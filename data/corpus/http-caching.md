---
title: HTTP caching
slug: Web/HTTP/Caching
page-type: guide
---

# HTTP caching

The HTTP cache stores a response so that later equivalent requests can reuse it.
Caching reduces latency and load, but a stale response served to a user is a
real bug, so the rules deserve care.

## Cache-Control

The `Cache-Control` header carries the caching policy. Directives apply to
requests and responses, and multiple directives are comma separated.

```http
Cache-Control: max-age=3600, must-revalidate
```

`max-age` gives the freshness lifetime in seconds. `no-cache` does not mean "do
not store": it means the cache must revalidate with the origin before reuse.
The directive that actually forbids storing a response is `no-store`, and
confusing the two is the most common caching mistake.

`private` restricts storage to a single user's browser cache, while `public`
allows shared caches such as a CDN to store the response. Any response carrying
per-user data must be marked `private`.

## Validators and conditional requests

When a stored response becomes stale, the cache revalidates instead of
re-downloading. A strong validator is `ETag`, and a weak one is
`Last-Modified`.

```http
GET /style.css HTTP/1.1
If-None-Match: "abc123"
```

If the representation is unchanged the origin answers `304 Not Modified` with no
body, and the cache serves its stored copy. This saves bandwidth even when the
freshness lifetime has already expired.

### Immutable assets and cache busting

Build pipelines usually embed a content hash in asset filenames, for example
`app.6f2c1a.js`. Because the URL changes whenever the content changes, the
response can be cached aggressively and never revalidated.

```http
Cache-Control: public, max-age=31536000, immutable
```

The HTML document that references those assets must NOT use a long lifetime,
because it is the file that points at the new hashed URLs. A typical setup pairs
a one-year lifetime for hashed assets with `no-cache` for the entry document.

## Stale responses while revalidating

`stale-while-revalidate` allows a cache to serve a stale response immediately
while it refreshes the entry in the background, which removes the revalidation
round trip from the critical path. `stale-if-error` allows a stale response to
be served when the origin is failing, which is a cheap availability improvement.

## Vary

The `Vary` header lists the request headers that participate in cache key
matching. A response that differs by `Accept-Encoding` must say so, otherwise a
cache may serve a gzip-encoded body to a client that did not ask for it.
Varying on a header with many values, such as `User-Agent`, fragments the cache
so badly that the hit rate collapses.
