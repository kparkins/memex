#!/bin/sh
set -e

echo "Starting MongoDB initialization..."

export MONGO_HOST=${MONGO_HOST:-mongod.search-community}
export MONGO_PORT=${MONGO_PORT:-27017}
export MONGOT_PASSWORD=${MONGOT_PASSWORD:-mongot}
export MONGODB_INITDB_ROOT_USERNAME=${MONGODB_INITDB_ROOT_USERNAME:-${MONGO_INITDB_ROOT_USERNAME:-admin}}
export MONGODB_INITDB_ROOT_PASSWORD=${MONGODB_INITDB_ROOT_PASSWORD:-${MONGO_INITDB_ROOT_PASSWORD:-admin}}

# Wait for MongoDB to be ready
echo "Waiting for MongoDB to be ready..."
until mongosh \
  --quiet \
  --host "$MONGO_HOST" \
  --port "$MONGO_PORT" \
  --username "$MONGODB_INITDB_ROOT_USERNAME" \
  --password "$MONGODB_INITDB_ROOT_PASSWORD" \
  --authenticationDatabase admin \
  --eval "db.adminCommand('ping')" > /dev/null 2>&1; do
  echo "Waiting for MongoDB..."
  sleep 2
done

echo "MongoDB is ready, proceeding with initialization..."

echo "Initializing replica set rs0..."
mongosh --quiet \
  --host "$MONGO_HOST" \
  --port "$MONGO_PORT" \
  --username "$MONGODB_INITDB_ROOT_USERNAME" \
  --password "$MONGODB_INITDB_ROOT_PASSWORD" \
  --authenticationDatabase admin \
  --eval '
    try {
      rs.initiate({
        _id: "rs0",
        members: [{ _id: 0, host: "mongod.search-community:27017" }]
      });
      print("Replica set initiated.");
    } catch (error) {
      if (error.codeName === "AlreadyInitialized") {
        print("Replica set already initialized.");
      } else {
        throw error;
      }
    }
  '

echo "Waiting for primary election..."
mongosh --quiet \
  --host "$MONGO_HOST" \
  --port "$MONGO_PORT" \
  --username "$MONGODB_INITDB_ROOT_USERNAME" \
  --password "$MONGODB_INITDB_ROOT_PASSWORD" \
  --authenticationDatabase admin \
  --eval '
    const deadline = Date.now() + 60000;
    let primary = null;

    while (Date.now() < deadline) {
      const status = rs.status();
      primary = status.members.find((member) => member.stateStr === "PRIMARY");
      if (primary) {
        print("Primary elected: " + primary.name);
        break;
      }
      sleep(1000);
    }

    if (!primary) {
      throw new Error("Timed out waiting for primary election.");
    }
  '

# Create mongot user
echo "Creating mongotUser..."
mongosh --quiet \
  --host "$MONGO_HOST" \
  --port "$MONGO_PORT" \
  --username "$MONGODB_INITDB_ROOT_USERNAME" \
  --password "$MONGODB_INITDB_ROOT_PASSWORD" \
  --authenticationDatabase admin \
  admin \
  --eval "
    try {
      db.createUser({
        user: 'mongotUser',
        pwd: '$MONGOT_PASSWORD',
        roles: [{ role: 'searchCoordinator', db: 'admin' }]
      });
      print('mongotUser created.');
    } catch (error) {
      if (error.codeName === 'DuplicateKey' || error.code === 51003) {
        print('mongotUser already exists, skipping.');
      } else {
        throw error;
      }
    }
  "

echo "MongoDB initialization completed successfully."
