# Model Context Protocol (MCP) Server

EpochDB includes a native **Model Context Protocol (MCP)** server, allowing AI coding assistants and desktop agents—including **Cursor**, **Claude Desktop**, and **Antigravity**—to access persistent memory and Knowledge Graph traversal.

---

## Installation

Install the MCP CLI extra:

```bash
pip install "epochdb[mcp]"
```

---

## Starting the MCP Server

You can run the MCP server directly via the console script:

```bash
epochdb-mcp --storage-dir ~/.epochdb_mcp --embedding-model all-MiniLM-L6-v2
```

---

## Configuration

### Cursor Setup (`~/.cursor/mcp.json`)
Add EpochDB to your Cursor MCP settings:

```json
{
  "mcpServers": {
    "epochdb": {
      "command": "epochdb-mcp",
      "args": [
        "--storage-dir", "/home/user/.epochdb_data",
        "--embedding-model", "all-MiniLM-L6-v2"
      ]
    }
  }
}
```

### Claude Desktop Setup (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "epochdb": {
      "command": "python",
      "args": [
        "-m", "epochdb.mcp_server",
        "--storage-dir", "/home/user/.epochdb_data"
      ]
    }
  }
}
```

---

## Tools Exposed via MCP

| Tool Name | Parameters | Description |
| :--- | :--- | :--- |
| **`remember`** | `text`, `triples`, `category` | Stores a verbatim memory atom with optional Knowledge Graph triples. |
| **`query`** | `query`, `k`, `expand_hops` | Performs 5-stage retrieval with Topic Lock and supersession resolution. |
| **`get_entity`** | `name` | Returns entity connections and historical chronological timeline. |
| **`entity_graph`** | `name`, `depth` | Generates a multi-hop visual graph network around an entity. |
| **`delete_memory`**| `memory_id`, `hard` | Soft or hard deletes an atom from active retrieval. |
