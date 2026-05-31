#cloud-env [![npm version](http://img.shields.io/npm/v/cloud-env.svg)](https://www.npmjs.org/package/cloud-env) [![Build Status](http://img.shields.io/travis/ryanj/cloud-env.svg)](https://travis-ci.org/ryanj/cloud-env)

**!!AUTOCONFIGURE JS CODE FOR USE ON ANY PLATFORM NODE!!**

[cloud-env](https://github.com/ryanj/cloud-env) provides a vendor-neutral interface for autoconfiguring your app, making it portable across all major cloud hosting platforms / deployment targets:

* [OpenShift](https://developers.openshift.com/languages/nodejs/environment-variables.html#listen)
* [Heroku](https://devcenter.heroku.com/articles/dynos#web-dynos)
* [Modulus](http://help.modulus.io/customer/portal/articles/1660583)
* [CloudFoundry](http://docs.cloudfoundry.org/devguide/deploy-apps/environment-variable.html#PORT)

It works by checking the system environment (`process.env`) for known configuration strings, normalizing the results into [a well-defined list](#configuration-strings).

[![npm stats](https://nodei.co/npm/cloud-env.png?downloads=true&stars=true)](https://www.npmjs.org/package/cloud-env)

[![Dependency Check](http://img.shields.io/david/ryanj/cloud-env.svg)](https://david-dm.org/ryanj/cloud-env) [![monthly downloads](http://img.shields.io/npm/dm/cloud-env.svg)](https://www.npmjs.org/package/cloud-env) [![license](http://img.shields.io/npm/l/cloud-env.svg)](https://www.npmjs.org/package/cloud-env)

## Installation

The resulting config object contains the configuration settings that `cloud-env` was able to detect - including the server `PORT` number and bind `IP` address:

``` js
  //npm install cloud-env
  var config = require('cloud-env')
```

See the [Configuration Strings](#configuration-strings) list for more information about the settings that this module will automatically resolve.

## Listen up
Make sure to pass `config.PORT` and `config.IP` to your server's `listen` function:

```js
app.listen(config.PORT, config.IP, function () {
  console.log("Listening on "+config.IP+", port "+config.PORT)
});
```

If host-provided configs are not found, local development defaults are returned - allowing you to configure once, and run anywhere.

### Provide your own defaults
Use `.get('KEYNAME', default_value)` to fetch keys by name, optionally providing your own default values:

```js
port = config.get('PORT', 8000)
bind_address = config.get('IP','127.0.0.1')
app.listen(port, bind_address, function () {
  console.log("Listening on " + bind_address + ", port " + port)
});
```

The above example will default to port `8000` instead of `8080`, and will attempt to bind on '127.0.0.0.1' instead of '0.0.0.0'.

## Configuration Strings
Reliable configuration settings for local dev AND for "the cloud":

config.NAME | DEFAULT | process.env.SOURCE_VARS 
--------------------|-----------|---------------
IP                  | 0.0.0.0 | OPENSHIFT_NODEJS_IP, BIND_IP 
PORT                | 8080  | OPENSHIFT_NODEJS_PORT, PORT
HOSTNAME            | localhost  | OPENSHIFT_APP_DNS, HOSTNAME 
APP_NAME            | APP_NAME  | OPENSHIFT_APP_NAME, APP_NAME
MONGODB_DB_URL      | mongodb://127.0.0.1:27017  | OPENSHIFT_MONGODB_DB_URL, MONGODB_DB_URL
MONGODB_DB_HOST      | 127.0.0.1  | OPENSHIFT_MONGODB_DB_HOST, MONGODB_DB_HOST
MONGODB_DB_PORT      | 27017  | OPENSHIFT_MONGODB_DB_PORT, MONGODB_DB_PORT
MONGODB_DB_USERNAME      | undefined  | OPENSHIFT_MONGODB_DB_USERNAME, MONGODB_DB_USERNAME
MONGODB_DB_PASSWORD      | undefined  | OPENSHIFT_MONGODB_DB_PASSWORD, MONGODB_DB_PASSWORD
POSTGRESQL_DB_URL   | postgresql://127.0.0.1:5432  | OPENSHIFT_POSTGRESQL_DB_URL, POSTGRESQL_DB_URL
POSTGRESQL_DB_HOST   | 127.0.0.1  | OPENSHIFT_POSTGRESQL_DB_HOST, POSTGRESQL_DB_HOST
POSTGRESQL_DB_PORT   | 5432  | OPENSHIFT_POSTGRESQL_DB_PORT, POSTGRESQL_DB_PORT
POSTGRESQL_DB_USERNAME   | undefined  | OPENSHIFT_POSTGRESQL_DB_USERNAME, POSTGRESQL_DB_USERNAME
POSTGRESQL_DB_PASSWORD   | undefined  | OPENSHIFT_POSTGRESQL_DB_PASSWORD, POSTGRESQL_DB_PASSWORD
MYSQL_DB_URL      | mysql://127.0.0.1:3306  | OPENSHIFT_MYSQL_DB_URL, MYSQL_DB_URL
MYSQL_DB_HOST      | 127.0.0.1  | OPENSHIFT_MYSQL_DB_HOST, MYSQL_DB_HOST
MYSQL_DB_PORT      | 3306  | OPENSHIFT_MYSQL_DB_PORT, MYSQL_DB_PORT
MYSQL_DB_USERNAME      | undefined  | OPENSHIFT_MYSQL_DB_USERNAME, MYSQL_DB_USERNAME
MYSQL_DB_PASSWORD      | undefined  | OPENSHIFT_MYSQL_DB_PASSWORD, MYSQL_DB_PASSWORD

### Advanced Configuration

See [`config-multipaas`](https://github.com/ryanj/config-multipaas/) and the related [`config-chain` API docs](https://github.com/dominictarr/config-chain/#boring-api-docs) for a more advanced configuration solution that incorporates the same set of cloud configuration keys.

![MultiPaaS](http://i.imgur.com/fCi6YX6.png)
