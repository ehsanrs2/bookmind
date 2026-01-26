# Bookmind

Bookmind scopes define how a request targets books, chapters, pages, and tags for retrieval. Chunk metadata describes the stored pieces of content used to answer those requests.

- Page reference policy: PDF page numbers are primary; book page numbers are optional when available.
- Combine logic: selectors use OR within each group (chapters, pages, tags) and AND between groups. Tag mode is ANY (match any tag) or ALL (match every tag).
- Policy note: book_qa requires citations; exam_generator does not by default.

Locations in this repo:
- Schemas: `backend/open_webui/schemas/bookmind/`
- Examples: `test/test_files/bookmind/`
- Validator: `backend/open_webui/utils/bookmind_validator.py`
