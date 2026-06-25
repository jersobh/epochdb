#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <liburing.h>
#include <sys/stat.h>
#include <sys/types.h>

#define WAL_BLOCK_SIZE 4096
#define ALLOCATION_CHUNK (10 * 1024 * 1024) // 10MB pre-allocation chunk

typedef struct {
    int fd;
    off_t write_offset;
    off_t allocated_size;
    struct io_uring ring;
    int has_sqpoll;
} UringWriter;

// Exported functions
UringWriter* uring_writer_open(const char* filepath);
int uring_writer_write(UringWriter* writer, const char* data, size_t length, int sync);
void uring_writer_reap_completions(UringWriter* writer);
void uring_writer_close(UringWriter* writer);

UringWriter* uring_writer_open(const char* filepath) {
    UringWriter* writer = malloc(sizeof(UringWriter));
    if (!writer) {
        return NULL;
    }
    memset(writer, 0, sizeof(UringWriter));

    // Open file using Direct I/O. If filesystem doesn't support O_DIRECT (like tmpfs), fall back to standard open.
    writer->fd = open(filepath, O_WRONLY | O_CREAT | O_DIRECT, 0644);
    if (writer->fd < 0) {
        writer->fd = open(filepath, O_WRONLY | O_CREAT, 0644);
        if (writer->fd < 0) {
            free(writer);
            return NULL;
        }
    }

    // Determine starting file offset (crucial if file already exists)
    struct stat st;
    if (fstat(writer->fd, &st) == 0) {
        writer->write_offset = st.st_size;
        writer->allocated_size = st.st_size;
    } else {
        writer->write_offset = 0;
        writer->allocated_size = 0;
    }

    // Pre-allocate the first block chunk if starting a new file
    if (writer->write_offset == 0) {
        if (posix_fallocate(writer->fd, 0, ALLOCATION_CHUNK) == 0) {
            writer->allocated_size = ALLOCATION_CHUNK;
        }
    }

    // Attempt to set up SQPOLL (kernel submission polling thread) for maximum throughput.
    struct io_uring_params params;
    memset(&params, 0, sizeof(params));
    params.flags = IORING_SETUP_SQPOLL;
    params.sq_thread_idle = 2000; // Idle timeout of 2000 ms

    int ret = io_uring_queue_init_params(64, &writer->ring, &params);
    if (ret == 0) {
        writer->has_sqpoll = 1;
    } else {
        // Fall back to standard io_uring if SQPOLL is restricted (e.g., lack of privileges on older systems)
        ret = io_uring_queue_init(64, &writer->ring, 0);
        if (ret != 0) {
            close(writer->fd);
            free(writer);
            return NULL;
        }
        writer->has_sqpoll = 0;
    }

    return writer;
}

int uring_writer_write(UringWriter* writer, const char* data, size_t length, int sync) {
    if (!writer || writer->fd < 0) {
        return -1;
    }

    // Pad length to nearest WAL_BLOCK_SIZE boundary for O_DIRECT compliance
    size_t padded_length = ((length + WAL_BLOCK_SIZE - 1) / WAL_BLOCK_SIZE) * WAL_BLOCK_SIZE;

    // Allocate page-aligned buffer for Direct Memory Access (DMA) compatibility
    void* buf = NULL;
    if (posix_memalign(&buf, WAL_BLOCK_SIZE, padded_length) != 0) {
        return -2;
    }

    // Copy original data and pad the rest with spaces
    memcpy(buf, data, length);
    if (padded_length > length) {
        memset((char*)buf + length, ' ', padded_length - length);
    }

    // Ensure space is pre-allocated
    if (writer->write_offset + (off_t)padded_length > writer->allocated_size) {
        off_t new_allocation = writer->allocated_size + ALLOCATION_CHUNK;
        if (posix_fallocate(writer->fd, writer->allocated_size, ALLOCATION_CHUNK) == 0) {
            writer->allocated_size = new_allocation;
        }
    }

    // Prepare SQE
    struct io_uring_sqe* sqe = io_uring_get_sqe(&writer->ring);
    if (!sqe) {
        // Submission queue is full. Submit pending SQEs and wait for at least one completion (backpressure)
        io_uring_submit(&writer->ring);
        struct io_uring_cqe* cqe;
        if (io_uring_wait_cqe(&writer->ring, &cqe) == 0) {
            void* completed_buf = io_uring_cqe_get_data(cqe);
            if (completed_buf) {
                free(completed_buf);
            }
            io_uring_cqe_seen(&writer->ring, cqe);
        }
        sqe = io_uring_get_sqe(&writer->ring);
        if (!sqe) {
            free(buf);
            return -3;
        }
    }

    io_uring_prep_write(sqe, writer->fd, buf, padded_length, writer->write_offset);
    
    // Store buffer pointer in user data to free it later upon completion
    io_uring_sqe_set_data(sqe, buf);

    int ret;
    if (sync) {
        // Submit and block until the write completes
        ret = io_uring_submit_and_wait(&writer->ring, 1);
        if (ret < 0) {
            free(buf);
            return ret;
        }

        // Reap the completion immediately
        struct io_uring_cqe* cqe;
        ret = io_uring_wait_cqe(&writer->ring, &cqe);
        if (ret == 0) {
            void* completed_buf = io_uring_cqe_get_data(cqe);
            if (completed_buf) {
                free(completed_buf);
            }
            io_uring_cqe_seen(&writer->ring, cqe);
        } else {
            // Wait failed, let's free buffer to avoid leak
            free(buf);
            return ret;
        }
    } else {
        // Submit asynchronously, buffer freed later in reap_completions
        ret = io_uring_submit(&writer->ring);
        if (ret < 0) {
            free(buf);
            return ret;
        }
    }

    writer->write_offset += padded_length;
    return 0;
}

void uring_writer_reap_completions(UringWriter* writer) {
    if (!writer) {
        return;
    }

    struct io_uring_cqe* cqe;
    // Non-blocking drain of completion queue
    while (io_uring_peek_cqe(&writer->ring, &cqe) == 0) {
        void* buf = io_uring_cqe_get_data(cqe);
        if (buf) {
            free(buf);
        }
        io_uring_cqe_seen(&writer->ring, cqe);
    }
}

void uring_writer_close(UringWriter* writer) {
    if (!writer) {
        return;
    }

    // Submit and wait for all outstanding I/O operations to complete
    struct io_uring_cqe* cqe;
    while (io_uring_sq_ready(&writer->ring) > 0 || io_uring_cq_ready(&writer->ring) > 0) {
        io_uring_submit_and_wait(&writer->ring, 1);
        while (io_uring_peek_cqe(&writer->ring, &cqe) == 0) {
            void* buf = io_uring_cqe_get_data(cqe);
            if (buf) {
                free(buf);
            }
            io_uring_cqe_seen(&writer->ring, cqe);
        }
    }

    io_uring_queue_exit(&writer->ring);

    if (writer->fd >= 0) {
        // Truncate file back to the actual written size (reclaims unused pre-allocated blocks)
        ftruncate(writer->fd, writer->write_offset);
        close(writer->fd);
    }

    free(writer);
}
