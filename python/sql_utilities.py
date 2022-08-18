import os, os.path
import pandas as pd
import sqlalchemy
from typing import Union


##  get a table from a database
def sql_table_to_df(
    engine: sqlalchemy.engine.Engine,
    table_name: str,
    fields_select: Union[list, str] = None,
    query_append: str = None
) -> pd.DataFrame:
    """
        Query a database, retrieve a table, and convert it to a dataframe.
        - engine: SQLalchemy Engine used to create a connection and query the database
        - table_name: the table in the database to query
        - fields_select: a list of fields to select or a string of fields to select (comma-delimited). If None, return all fields.
        - query_append: any restrictions to place on the query (e.g., where). If None, return all records.
    """

    # check table names
    if table_name not in engine.table_names():
        # LOGHERE
        return None

    # build the query
    if fields_select is not None:
        fields_select_str = ", ".join(fields_select) if isinstance(fields_select, list) else fields_select
    else:
        fields_select_str = "*"
    query_append = "" if (query_append is None) else f" {query_append}"
    query = f"select {fields_select_str} from {table_name}{query_append};"

    # try the connection
    with engine.connect() as con:
        try:
            df_out = pd.read_sql_query(query, con)
        except Exception as e:
            # LOGHERE
            raise RuntimeError(f"Error in sql_table_to_df: the service returned error '{e}'.\n\nQuery:\n\t'{query}'.")

    return df_out



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
            except Exception as e:
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
