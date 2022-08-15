import os, os.path
import pandas as pd
import sqlalchemy
from typing import Union

##  using a dictionary, write dataframes to a database using a connection
def _write_dataframes_to_db(
    dict_tables: dict,
    db_engine: Union[sqlalchemy.engine.Engine, str],
    preseve_table_schema: bool = True
):
    """
        write a dictionary of tables to a data base
        - dict_tables: dictionary of form {TABLENAME: pd.DataFrame, ...} used to write the table to the database
        - db_engine: an existing SQLAlchemy database engine or a file path to an SQLite database used to establish a connection
            * If a file path is specified, the connection will be opened and closed within the function
        - preseve_table_schema: preserve existing schema? If so, before writing new tables, rows in existing tables will be deleted and the table will be appended.
    """

    # check input specification
    if isinstance(db_engine, str):
        if os.path.exists(db_engine) and db_engine.endswith(".sqlite"):
            try:
                db_engine = sqlalchemy.create_engine(f"sqlite:///{db_engine}")
            except e:
                raise ValueError(f"Error establishing a connection to sqlite database at {db_engine}: {e} ")
    elif not isinstance(db_engine, sqlalchemy.engine.Engine):
        t = type(db_engine)
        raise ValueError(f"Invalid db_con type {t}: only types str, sqlite3.Connection are valid.")

    # get available tables
    tables_avail = db_engine.table_names()
    with db_engine.connect() as con:
        for table in dict_tables.keys():
            if table in tables_avail:
                if preseve_table_schema:
                    # first, execute a query to remove all rows--we want to preserve the schema that's set up by Julia
                    con.execute(f"delete from {table};")
                    dict_tables[table].to_sql(table, con, if_exists = "append", index = None)
                else:
                    dict_tables[table].to_sql(table, con, if_exists = "replace", index = None)
