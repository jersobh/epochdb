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
| **`epochdb_remember`** | `text`, `metadata`, `memory_type` | Stores a verbatim memory atom (`general`, `episodic`, `profile`, `working`, `skill`). |
| **`epochdb_query`** | `query`, `k`, `min_score`, `memory_type`, `context_window` | Hybrid retrieval (semantic + keyword + KG) with Topic Lock and supersession. |
| **`epochdb_multi_hop`** | `query`, `hops`, `k`, `context_window` | Multi-hop relational search across the entity graph. |
| **`epochdb_adaptive_query`** | `query`, `k`, `context_window` | Routes to semantic / relational / temporal / quantitative engines. |
| **`epochdb_get_timeline`** | `entity_id`, `start`, `end` | Chronological history for an entity (or all memories). |
| **`epochdb_entity_graph`** | `entity_id`, `depth` | Multi-hop graph neighborhood around an entity. |
| **`epochdb_update`** | `memory_id`, `text`, `metadata` | Update an existing atom. |
| **`epochdb_delete`** | `memory_id`, `hard` | Soft or hard delete. |
| **`epochdb_analyze`** | `text` | Extract `(subject, predicate, object)` triples without storing. |
| **`epochdb_remember_skill`** | `skill_name`, `description`, `steps`, … | Store a procedural `MemoryType.SKILL` atom. |
| **`epochdb_get_skill`** | `skill_name` | Resolve a skill by name / id via the skill index. |
| **`epochdb_list_skills`** | — | List all stored skills. |
| **`epochdb_remember_user_profile`** | `user_id`, `fact_text`, `metadata` | Store a long-term profile fact. |
| **`epochdb_get_user_profile`** | `user_id` | Retrieve profile facts for a user. |
| **`epochdb_get_hot_summary_snapshot`** | `user_id` | Compact profile + skills block for system-prompt injection. |
