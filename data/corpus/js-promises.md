---
title: Using promises
slug: Web/JavaScript/Guide/Using_promises
page-type: guide
---

# Using promises

A promise is an object representing the eventual completion or failure of an
asynchronous operation. A promise is in one of three states: pending, fulfilled,
or rejected. Once it is fulfilled or rejected it is settled, and its state can
never change again.

## Chaining

The `then()` method returns a new promise, which is what makes chaining work.
The value returned from a callback becomes the fulfillment value of the next
promise in the chain.

```js
fetch("/api/user")
  .then((response) => response.json())
  .then((user) => console.log(user.name));
```

A classic mistake is forgetting to return the inner promise, which breaks the
chain and causes the next callback to receive `undefined` before the inner
operation has finished.

## Error handling

A rejection propagates down the chain until it meets a handler. `catch()` is
shorthand for `then(undefined, onRejected)`, and it also catches exceptions
thrown synchronously inside earlier callbacks.

```js
try {
  const response = await fetch("/api/user");
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
} catch (error) {
  console.error(error);
}
```

### Why fetch does not reject on 404

`fetch()` rejects only on network failure or a request that could not complete.
An HTTP error status such as 404 or 500 is still a successful HTTP exchange, so
the promise fulfills. Checking `response.ok` explicitly is therefore required,
and its absence is one of the most common bugs in browser code.

## Composition

`Promise.all()` waits for every promise to fulfill and rejects immediately if
any of them rejects. `Promise.allSettled()` always waits for all of them and
reports each outcome. `Promise.race()` settles as soon as the first promise
settles, and `Promise.any()` waits for the first fulfillment, ignoring earlier
rejections.

```js
const [profile, settings] = await Promise.all([
  fetch("/api/profile").then((r) => r.json()),
  fetch("/api/settings").then((r) => r.json()),
]);
```

Running independent requests with `Promise.all()` instead of sequential `await`
statements is usually the cheapest available latency win, because the requests
overlap instead of queueing.

## Async functions and await

An `async` function always returns a promise. `await` pauses the function until
the awaited promise settles, and unwraps its value.

Awaiting inside a loop serializes the work. When the iterations are independent,
map them to promises first and await the whole collection.

## Common pitfalls

The most frequent error is a floating promise: calling an async function without
awaiting it or attaching a handler, so a rejection becomes an unhandled promise
rejection. Executor functions passed to `new Promise` should also avoid async
work whose errors cannot be routed to `reject`.
