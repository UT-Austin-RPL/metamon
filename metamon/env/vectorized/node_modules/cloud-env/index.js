var defaults = {
  dev: {
    PORT: 8080,
    IP: '0.0.0.0',
    HOSTNAME: 'localhost',
    APP_NAME: 'APP_NAME',
    MONGODB_DB_URL: 'mongodb://127.0.0.1:27017',
    MONGODB_DB_HOST: '127.0.0.1',
    MONGODB_DB_PORT: 27017,
    MONGODB_DB_USERNAME: undefined,
    MONGODB_DB_PASSWORD: undefined,
    POSTGRESQL_DB_URL: 'postgresql://127.0.0.1:5432',
    POSTGRESQL_DB_HOST: '127.0.0.1',
    POSTGRESQL_DB_PORT: 5432,
    POSTGRESQL_DB_USERNAME: undefined,
    POSTGRESQL_DB_PASSWORD: undefined,
    MYSQL_DB_URL:  'mysql://127.0.0.1:3306',
    MYSQL_DB_HOST: '127.0.0.1',
    MYSQL_DB_PORT: 3306,
    MYSQL_DB_USERNAME: undefined,
    MYSQL_DB_PASSWORD: undefined,
  },
  cloud: {
    PORT: process.env.PORT,
    IP: process.env.BIND_IP,
    HOSTNAME: process.env.HOSTNAME,
    APP_NAME: process.env.APP_NAME,
    MONGODB_DB_URL: process.env.MONGODB_DB_URL,
    MONGODB_DB_HOST: process.env.MONGODB_DB_HOST,
    MONGODB_DB_PORT: process.env.MONGODB_DB_PORT,
    MONGODB_DB_USERNAME: process.env.MONGODB_DB_USERNAME,
    MONGODB_DB_PASSWORD: process.env.MONGODB_DB_PASSWORD,
    POSTGRESQL_DB_URL: process.env.POSTGRESQL_DB_URL,
    POSTGRESQL_DB_HOST: process.env.POSTGRESQL_DB_HOST,
    POSTGRESQL_DB_PORT: process.env.POSTGRESQL_DB_PORT,
    POSTGRESQL_DB_USERNAME: process.env.POSTGRESQL_DB_USERNAME,
    POSTGRESQL_DB_PASSWORD: process.env.POSTGRESQL_DB_PASSWORD,
    MYSQL_DB_URL:  process.env.MYSQL_DB_URL,
    MYSQL_DB_HOST: process.env.MYSQL_DB_HOST,
    MYSQL_DB_PORT: process.env.MYSQL_DB_PORT,
    MYSQL_DB_USERNAME: process.env.MYSQL_DB_USERNAME,
    MYSQL_DB_PASSWORD: process.env.MYSQL_DB_PASSWORD,
  },
  openshift: {
    PORT: process.env.OPENSHIFT_NODEJS_PORT,
    IP: process.env.OPENSHIFT_NODEJS_IP,
    HOSTNAME: process.env.OPENSHIFT_APP_DNS,
    APP_NAME: process.env.OPENSHIFT_APP_NAME,
    MONGODB_DB_URL: process.env.OPENSHIFT_MONGODB_DB_URL,
    MONGODB_DB_HOST: process.env.OPENSHIFT_MONGODB_DB_HOST,
    MONGODB_DB_PORT: process.env.OPENSHIFT_MONGODB_DB_PORT,
    MONGODB_DB_USERNAME: process.env.OPENSHIFT_MONGODB_DB_USERNAME,
    MONGODB_DB_PASSWORD: process.env.OPENSHIFT_MONGODB_DB_PASSWORD,
    POSTGRESQL_DB_URL: process.env.OPENSHIFT_POSTGRESQL_DB_URL,
    POSTGRESQL_DB_HOST: process.env.OPENSHIFT_POSTGRESQL_DB_HOST,
    POSTGRESQL_DB_PORT: process.env.OPENSHIFT_POSTGRESQL_DB_PORT,
    POSTGRESQL_DB_USERNAME: process.env.OPENSHIFT_POSTGRESQL_DB_USERNAME,
    POSTGRESQL_DB_PASSWORD: process.env.OPENSHIFT_POSTGRESQL_DB_PASSWORD,
    MYSQL_DB_URL: process.env.OPENSHIFT_MYSQL_DB_URL,
    MYSQL_DB_HOST: process.env.OPENSHIFT_MYSQL_DB_HOST,
    MYSQL_DB_PORT: process.env.OPENSHIFT_MYSQL_DB_PORT,
    MYSQL_DB_USERNAME: process.env.OPENSHIFT_MYSQL_DB_USERNAME,
    MYSQL_DB_PASSWORD: process.env.OPENSHIFT_MYSQL_DB_PASSWORD
  },
  v3: {
    APP_NAME: process.env.OPENSHIFT_BUILD_NAMESPACE,
    MONGODB_DB_HOST: process.env.MONGODB_SERVICE_HOST,
    MONGODB_DB_PORT: process.env.MONGODB_SERVICE_PORT,
    MONGODB_DB_USERNAME: process.env.MONGODB_USER,
    MONGODB_DB_PASSWORD: process.env.MONGODB_PASSWORD,
    POSTGRESQL_DB_HOST: process.env.POSTGRESQL_SERVICE_HOST,
    POSTGRESQL_DB_PORT: process.env.POSTGRESQL_SERVICE_PORT,
    POSTGRESQL_DB_USERNAME: process.env.POSTGRESQL_USER,
    POSTGRESQL_DB_PASSWORD: process.env.POSTGRESQL_PASSWORD,
    MYSQL_DB_HOST: process.env.MYSQL_SERVICE_HOST,
    MYSQL_DB_PORT: process.env.MYSQL_SERVICE_PORT,
    MYSQL_DB_USERNAME: process.env.MYSQL_USER,
    MYSQL_DB_PASSWORD: process.env.MYSQL_PASSWORD
  }
}

var findComponents = function(){
  var component_list = [];
  var name = '';
  for( var key in process.env ){
    name = key.replace('COMPONENT_','').replace('_HOST','')
    if(key.indexOf('COMPONENT_'+name+'_HOST') == 0 && process.env['COMPONENT_'+name+'_PORT']){
      component_list.push(name);
    }
  }
  return component_list;
}

var resolve_config = function (){
  var env = {};
  var self_svc_key = false;
  var db_names     = ['postgresql','mongodb','mysql'];
  var db_key       = '';
  var svc_name     = process.env.OPENSHIFT_BUILD_NAME;

  //kubernetes and openshiftV3 configs
  // Detect service host and port from v3 build namespace
  if(svc_name){
    // strip build number
    self_svc_key = svc_name.replace(/-\d+$/,'').toUpperCase().replace(/-/g, '_');
    defaults.v3['PORT'] = process.env[self_svc_key+'_SERVICE_PORT'];
  }
  // Sniff common v3 database configs
  for(var db in db_names){
    db_key = db_names[db].toUpperCase();
    if( process.env[db_key+'_USER'] && process.env[db_key+'_PASSWORD'] && process.env[db_key+'_SERVICE_HOST'] && process.env[db_key+'_SERVICE_PORT'] )
    {
      defaults.v3[db_key+'_DB_URL'] = db_names[db]+"://"+process.env[db_key+'_USER']+":"+process.env[db_key+'_PASSWORD']+"@"+process.env[db_key+'_SERVICE_HOST']+":"+process.env[db_key+'_SERVICE_PORT'];
    }
  }

  env.defaults   = defaults;
  env.components = findComponents();
  env.component  = {};

  for(var cmp in env.components){
    env.component[env.components[cmp]] = {
      host: process.env['COMPONENT_'+env.components[cmp]+'_HOST'],
      port: process.env['COMPONENT_'+env.components[cmp]+'_PORT']
    }
  }

  // config key accessor
  env.get = function (key, default_key){
    return defaults.v3[key] || defaults.openshift[key] || defaults.cloud[key] || default_key || defaults.dev[key];
  }

  // collapse / unify available configs
  for(var key in defaults.cloud){
    env[key] = env.get(key);
  }
  return env;
}

var exports = module.exports = resolve_config();
