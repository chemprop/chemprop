DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS ckpt;
DROP TABLE IF EXISTS model;
DROP TABLE IF EXISTS dataset;

CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  preferences JSON
);

CREATE TABLE ckpt (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ckpt_name TEXT NOT NULL,
  associated_user INTEGER NOT NULL,
  created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  class TEXT NOT NULL,
  stats JSON,
  epochs INTEGER NOT NULL DEFAULT 30,
  ensemble_size INTEGER NOT NULL,
  training_size INTEGER NOT NULL,
  completed BOOLEAN NOT NULL DEFAULT 0,
  FOREIGN KEY (associated_user) REFERENCES user (id),
  CONSTRAINT uq_ckpt_names UNIQUE(ckpt_name, associated_user)
);

CREATE TABLE model (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  associated_ckpt INTEGER NOT NULL,
  FOREIGN KEY (associated_ckpt) REFERENCES ckpt (id)
);

CREATE TABLE dataset (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_name TEXT NOT NULL,
  associated_user INTEGER NOT NULL,
  created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  class TEXT NOT NULL,
  FOREIGN KEY (associated_user) REFERENCES user (id),
  CONSTRAINT uq_dataset_names UNIQUE(dataset_name, associated_user)
);

INSERT INTO user (username) VALUES ("DEFAULT")