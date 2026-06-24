"""VioLane web application package (FastAPI routes + helpers).

The UI behavior is identical to the previous single-file ``web_app.py``; this
package only reorganizes that module into focused units:

- ``runtime``           shared app state (templates, in-memory job/stage stores)
- ``media``             frame/thumbnail encoding + summary serialization
- ``pipeline_factory``  rule expansion, pipeline construction, full-pass runner
- ``routes_*``          FastAPI routers grouped by concern
"""
