# Bookmind <> Open WebUI Integration Map

_Last updated: 2026-02-01_

## Backend (FastAPI + services)
- `backend/open_webui/main.py` - FastAPI entrypoint; wires routers and defines `/api/chat/completions` (and `/api/v1/chat/completions`) used by the UI.
- `backend/open_webui/routers/openai.py` - OpenAI-compatible `/chat/completions` handler; applies model params + access checks and forwards to upstream providers.
- `backend/open_webui/utils/middleware.py` - Chat pipeline: `process_chat_payload`, `chat_completion_files_handler`, `chat_memory_handler`, `chat_web_search_handler`, `chat_completion_tools_handler`, `process_chat_response`.
- `backend/open_webui/socket/main.py` - Socket events for chat streaming, task progress, and client updates.
- `backend/open_webui/routers/knowledge.py` - Knowledge base CRUD, ACL, file association, reindex, export; embeds KB metadata into vector DB.
- `backend/open_webui/routers/retrieval.py` - Ingestion + RAG endpoints (`process_file`, `process_files_batch`, web/youtube ingestion, web search).
- `backend/open_webui/retrieval/utils.py` - RAG utilities (embedding/reranking adapters, `get_sources_from_items`, query helpers).
- `backend/open_webui/retrieval/vector/factory.py` + `backend/open_webui/retrieval/vector/dbs/*` - Vector DB abstraction and concrete drivers; `VECTOR_DB_CLIENT` singleton.
- `backend/open_webui/routers/memories.py` - User memory CRUD backed by vector collections.
- `backend/open_webui/models/chats.py` - Chat storage (`chat` JSON) + `chat_file` join table.
- `backend/open_webui/models/files.py` - File metadata/content model used by KBs and chat attachments.
- `backend/open_webui/models/knowledge.py` - Knowledge base + `knowledge_file` join table + ACL metadata.
- `backend/open_webui/models/users.py` - User and API key models.
- `backend/open_webui/utils/access_control.py` - ACL checks (`has_access`, `has_permission`, etc.).
- `backend/open_webui/utils/auth.py` + `backend/open_webui/routers/auths.py` - Token auth, session user, OAuth/LDAP flows.
- `backend/open_webui/internal/db.py` - SQLAlchemy setup + peewee migration hook.
- `backend/open_webui/migrations/*` - Alembic migrations.
- `backend/open_webui/internal/migrations/*` - Peewee migrations run before Alembic (see `internal/db.py`).

## Frontend (SvelteKit)
### Chat entry + request flow
- `src/routes/(app)/c/[id]/+page.svelte` - Chat route wrapper.
- `src/lib/components/chat/Chat.svelte` - Main chat controller; builds payload with `messages`, `files`, `features`, `tool_ids`, `filter_ids`, `background_tasks`, `session_id`, `chat_id` and submits to API.
- `src/lib/apis/openai/index.ts` - `generateOpenAIChatCompletion()` POSTs to `/api/chat/completions`; `chatCompletion()` used for direct provider streaming.
- `src/routes/+layout.svelte` - Socket event hub; handles `request:chat:completion` and forwards streaming responses to the socket channel.

### Knowledge UI
- `src/routes/(app)/workspace/knowledge/+page.svelte` - Knowledge base list.
- `src/lib/components/workspace/Knowledge.svelte` - KB search/list and actions (export/delete).
- `src/routes/(app)/workspace/knowledge/create/+page.svelte` + `src/lib/components/workspace/Knowledge/CreateKnowledgeBase.svelte` - KB creation + ACL UI.
- `src/routes/(app)/workspace/knowledge/[id]/+page.svelte` + `src/lib/components/workspace/Knowledge/KnowledgeBase.svelte` - KB detail, file management, web/youtube ingestion.
- `src/lib/apis/knowledge/*` + `src/lib/apis/retrieval/*` - API calls for KB CRUD and ingestion endpoints.

## Request-flow diagram (short)
```
UI Chat.svelte
  -> POST /api/chat/completions (JSON payload)
    -> main.py:chat_completion
      -> utils/middleware.py:process_chat_payload
        -> (optional) tools/memory/web_search/files handlers
        -> retrieval/utils.py:get_sources_from_items
      -> routers/openai.py:/chat/completions (if OpenAI-compatible provider)
      -> utils/middleware.py:process_chat_response
    -> socket events (chat:completion, chat:tags, chat:title)
```

## Integration points (exact functions/classes/routes)
- `/api/chat/completions` -> `backend/open_webui/main.py:chat_completion`
  - Request payload (UI -> server, built in `src/lib/components/chat/Chat.svelte`):
    ```json
    {
      "stream": true,
      "model": "model-id",
      "messages": [{"role":"user","content":"..."}],
      "params": {"stream_response": true, "stop": ["..."]},
      "files": [{"id":"file-id","type":"file"}],
      "filter_ids": ["filter-id"],
      "tool_ids": ["server:mcp:...","tool-id"],
      "tool_servers": [{"id":"server-id","specs":[...]}],
      "features": {"memory":true,"web_search":false,"image_generation":false,"code_interpreter":false,"voice":false},
      "variables": {"user_name":"...","user_location":"..."},
      "model_item": {"id":"model-id","direct":false},
      "session_id": "socket-id",
      "chat_id": "chat-id",
      "id": "message-id",
      "parent_id": "parent-message-id",
      "parent_message": {"id":"parent-message-id","content":"..."},
      "background_tasks": {"title_generation":true,"tags_generation":true,"follow_up_generation":true},
      "stream_options": {"include_usage": true}
    }
    ```
  - Server augments into `metadata` in `chat_completion` (chat_id/message_id/session_id/files/tool ids/etc) before pipeline.
- `/api/v1/chat/completions` (OpenAI-compat) -> `backend/open_webui/routers/openai.py:generate_chat_completion`
  - OpenAI-style payload; server filters/overrides params based on model config and forwards to provider `/chat/completions`.
- `process_chat_payload` -> `backend/open_webui/utils/middleware.py:process_chat_payload`
  - Entry for memory/web search/tools/files injection and RAG context assembly.
- File/RAG injection -> `backend/open_webui/utils/middleware.py:chat_completion_files_handler`
  - Reads `metadata.files`, runs query generation, fetches sources via retrieval utils.
- RAG retrieval -> `backend/open_webui/retrieval/utils.py:get_sources_from_items`
  - Inputs: `items` (files/notes/chats/urls/collections), `queries`, embedding fn, reranker fn.
- Knowledge CRUD + ACL -> `backend/open_webui/routers/knowledge.py` + `backend/open_webui/models/knowledge.py`
  - Create KB: `POST /api/v1/knowledge/create` payload:
    ```json
    { "name": "KB name", "description": "KB description", "access_control": {"read": {...}, "write": {...}} }
    ```
  - Add file to KB: `POST /api/v1/knowledge/{id}/file/add` payload:
    ```json
    { "file_id": "file-id" }
    ```
  - Update file in KB: `POST /api/v1/knowledge/{id}/file/update` payload:
    ```json
    { "file_id": "file-id" }
    ```
  - Remove file from KB: `POST /api/v1/knowledge/{id}/file/remove` payload:
    ```json
    { "file_id": "file-id" }
    ```
- File ingestion -> `backend/open_webui/routers/retrieval.py:process_file` / `process_files_batch`
  - Process file: `POST /api/v1/retrieval/process/file` payload:
    ```json
    { "file_id": "file-id", "content": "optional text", "collection_name": "optional-collection" }
    ```
  - Batch: `POST /api/v1/retrieval/process/files/batch` payload is a list of the same form.
- Vector client -> `backend/open_webui/retrieval/vector/factory.py:VECTOR_DB_CLIENT`
- ACL checks -> `backend/open_webui/utils/access_control.py`
- Auth user context -> `backend/open_webui/utils/auth.py:get_verified_user`
- Frontend chat submit -> `src/lib/components/chat/Chat.svelte` -> `src/lib/apis/openai/index.ts:generateOpenAIChatCompletion`
- Socket streaming -> `src/routes/+layout.svelte` + `backend/open_webui/socket/main.py`

## Vector store / document ingestion utilities
- Vector DB clients: `backend/open_webui/retrieval/vector/dbs/*` (pgvector, qdrant, milvus, pinecone, chroma, opensearch, weaviate, etc.).
- Ingestion pipeline: `backend/open_webui/routers/retrieval.py` (file/web/youtube ingestion) + `backend/open_webui/retrieval/loaders/*` (document loaders).
- RAG retrieval + reranking: `backend/open_webui/retrieval/utils.py` + `backend/open_webui/retrieval/models/*`.
- File storage providers for ingestion: `backend/open_webui/storage/provider.py`.
