# permessage-deflate [![Build status](https://secure.travis-ci.org/faye/permessage-deflate-node.svg)](http://travis-ci.org/faye/permessage-deflate-node)

Implements the
[permessage-deflate](https://tools.ietf.org/html/draft-ietf-hybi-permessage-compression)
WebSocket protocol extension as a plugin for
[websocket-extensions](https://github.com/faye/websocket-extensions-node).

## Installation

```
$ npm install permessage-deflate
```

## Usage

Add the plugin to your extensions:

```js
var Extensions = require('websocket-extensions'),
    deflate    = require('permessage-deflate');

var exts = new Extensions();
exts.add(deflate);
```

The extension can be configured, for example:

```js
var Extensions = require('websocket-extensions'),
    deflate    = require('permessage-deflate'),
    zlib       = require('zlib');

deflate = deflate.configure({
  level: zlib.Z_BEST_COMPRESSION,
  maxWindowBits: 13
});

var exts = new Extensions();
exts.add(deflate);
```

The set of available options can be split into two sets: those that control the
session's compressor for outgoing messages and do not need to be communicated to
the peer, and those that are negotiated as part of the protocol. The settings
only affecting the compressor are described fully in the [zlib
documentation](http://nodejs.org/api/zlib.html#zlib_options):

- `level`: sets the compression level, can be an integer from `0` to `9`, or one
  of the constants `zlib.Z_NO_COMPRESSION`, `zlib.Z_BEST_SPEED`,
  `zlib.Z_BEST_COMPRESSION`, or `zlib.Z_DEFAULT_COMPRESSION`
- `memLevel`: sets how much memory the compressor allocates, can be an integer
  from `1` to `9`, or one of the constants `zlib.Z_MIN_MEMLEVEL`,
  `zlib.Z_MAX_MEMLEVEL`, or `zlib.Z_DEFAULT_MEMLEVEL`
- `strategy`: can be one of the constants `zlib.Z_FILTERED`,
  `zlib.Z_HUFFMAN_ONLY`, `zlib.Z_RLE`, `zlib.Z_FIXED`, or
  `zlib.Z_DEFAULT_STRATEGY`

The other options relate to settings that are negotiated via the protocol and
can be used to set the local session's behaviour and control that of the peer:

- `noContextTakeover`: if `true`, stops the session reusing a deflate context
  between messages
- `requestNoContextTakeover`: if `true`, makes the session tell the other peer
  not to reuse a deflate context between messages
- `maxWindowBits`: an integer from `8` to `15` inclusive that sets the maximum
  size of the session's sliding window; a lower window size will be used if
  requested by the peer
- `requestMaxWindowBits`: an integer from `8` to `15` inclusive to ask the other
  peer to use to set its maximum sliding window size, if supported
