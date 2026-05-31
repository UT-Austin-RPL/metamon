### 0.1.7 / 2019-06-10

- Use the `Buffer.alloc()` and `Buffer.from()` functions instead of the unsafe
  `Buffer()` constructor

### 0.1.6 / 2017-09-10

- Use `9` instead of `8` as the `windowBits` parameter to zlib, to deal with
  restrictions introduced in zlib v1.2.9

### 0.1.5 / 2016-02-24

- Catch errors thrown by `close()` on zlib streams

### 0.1.4 / 2015-11-06

- The server does not send `server_max_window_bits` if the client does not ask
  for it; this works around an issue in Firefox.

### 0.1.3 / 2015-04-10

- Fix a race condition causing some fragments of deflate output to be dropped
- Make sure to emit minimal output on all Node versions

### 0.1.2 / 2014-12-18

- Don't allow configure() to be called with unrecognized options

### 0.1.1 / 2014-12-15

- Fix race condition when using context takeover, where adjacent messages have
  data listeners bound at the same time and end up duplicating output
- Use `DeflateRaw.flush()` correctly on v0.10 so that optimal compression is
  achieved

### 0.1.0 / 2014-12-13

- Initial release
