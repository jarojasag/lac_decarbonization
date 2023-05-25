import numpy as np
import os, os.path
import pandas as pd
import sqlalchemy
from typing import *



def dict_subset_to_query_append(
    dict_subset: Union[Dict[str, List], None],
    query_logic: str = "and"
) -> str:
    """
    Convert a subsetting dictionary to a "where" clause in an SQL query.

    Function Arguments
    ------------------
    - dict_subset: dictionary with keys that are columns in the table and 
        values, given as a list, to subset the table. dict_subset is written as:

        dict_subset = {
            field_a = [val_a1, val_a2, ..., val_am],
            field_b = [val_b1, val_b2, ..., val_bn],
            .
            .
            .
        }

    Keyword Arguments
    -----------------
    - query_logic: default is "and". Subsets table to as

        where field_a in (val_a1, val_a2, ..., val_am) ~ field_b in 
        (val_b1, val_b2, ..., val_bn)...

        where `~ in ["and", "or"]`

    """

    if dict_subset is None:
        return None

    # initialize query string and list used to build query string
    val_list = []

    for k in dict_subset.keys():
        vals = dict_subset.get(k)
        if vals is not None:
            val_str = join_list_for_query(vals) if isinstance(vals, list) else join_list_for_query([vals])
            val_str = f"{k} in ({val_str})" if isinstance(vals, list) else f"{k} = {val_str}"
            val_list.append(val_str)

    # only return a string if there are values to filter on
    query_str = ""
    if (len(val_list) > 0):
        query_str = f" {query_logic} ".join(val_list)
        query_str = f" where {query_str}"

    return query_str



def format_listlike_elements_for_filter_query(
    elements: Union[List, Tuple, np.ndarray, None],
    fields: Union[List, Tuple, np.ndarray, None],
    query_logic: str = "and"
) -> str:
    """
    Format a list-like set of elements for a filtering query
        using filtering fields and associated elements. Creates
        a query of the form

        (fields_0 = elements_0 ~ fields_1 = elements_1 ~ ...),

        where ~ is "and" or "or"

        * NOTE: if len(fields) != len(elements), reduces both
            to the minimum length available between the two.

    Function Arguments
    ------------------
    - elements: ordered elements to filter on, i.e.,

        [elements_0, elements_1, ..., elements_n]

    - fields: fields to use for filtering, i.e.,
        [fields_0, fields_1, ..., fields_n]

    Optional Arguments
    ------------------
    - query_logic: "and" or "or", used to define the query
    """
    # some checks
    if (elements is None) or (fields is None):
        return ""

    query_logic = "and" if not (query_logic in ["and", "or"]) else query_logic
    query = []

    for elem in elements:
        n = min(len(elem), len(fields))

        elem_cur = elem[0:n]
        fields_cur = fields[0:n]

        query_component = f" {query_logic} ".join([f"{x} = {format_type_for_sql_query(y)}" for x, y in zip(fields_cur, elem_cur)])
        query_component = f"({query_component})"
        query.append(query_component)

    query = " or ".join(query) if (len(query) > 0) else ""

    return query



def format_type_for_sql_query(
    val: Union[float, int, str]
) -> str:
    """
    Format values based on input type. If val is a string, adds quotes
    """

    val = f"'{val}'" if isinstance(val, str) else str(val)

    return val



def join_list_for_query(
    list_in: list,
    delim: str = ", "
) -> str:
    """
    Join the elements of a list to format for a query.

    Function Arguments
    ------------------
    - list_in: list of elements to format for query
        * If elements are strings, then adds ''
        * Otherwise, enter elements
    """

    list_join = [
        f"'{x}'" if isinstance(x, str) else str(x)
        for x in list_in
    ]

    return delim.join(list_join)



def sql_table_to_df(
    engine: sqlalchemy.engine.Engine,
    table_name: str,
    fields_select: Union[list, str] = None,
    query_append: str = None
) -> pd.DataFrame:
    """
    Query a database, retrieve a table, and convert it to a dataframe.

    Function Arguments
    ------------------
    - engine: SQLalchemy Engine used to create a connection and query the 
        database
    - table_name: the table in the database to query

    Keyword Arguments
    -----------------
    - fields_select: a list of fields to select or a string of fields to 
        select (comma-delimited). If None, return all fields.
    - query_append: any restrictions to place on the query (e.g., where). If 
        None, return all records.
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



def _write_dataframes_to_db(
    dict_tables: dict,
    db_engine: Union[sqlalchemy.engine.Engine, str],
    preserve_table_schema: bool = True,
    append_q: bool = False
) -> None:
    """
    Write a dictionary of tables to an SQL database.

    Function Arguments
    ------------------
    - dict_tables: dictionary of form {TABLENAME: pd.DataFrame, ...} used to 
        write the table to the database
    - db_engine: an existing SQLAlchemy database engine or a file path to an 
        SQLite database used to establish a connection
        * If a file path is specified, the connection will be opened and closed 
            within the function

    Keyword Arguments
    -----------------
    - preserve_table_schema: preserve existing schema? If so, before writing new 
        tables, rows in existing tables will be deleted and the table will be 
        appended.
    - append_q: set to True top append tables to existing tables if they exist 
        in the database
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
        raise ValueError(f"Invalid db_con type {t}: only types str, sqlalchemy.engine.Engine are valid.")

    # get available tables
    tables_avail = db_engine.table_names()
    with db_engine.connect() as con:
        for table in dict_tables.keys():
            #
            df_write = dict_tables.get(table)

            if table in tables_avail:

                # try retrieving columns
                df_columns = pd.read_sql_query(f"select * from {table} limit 0;", con)

                # initialize writing based on appendage/preserving schema
                cols_write = list(df_columns.columns)
                on_exists = "append"
                query = None
                write_q = set(df_columns.columns).issubset(set(df_write.columns))

                if not append_q:
                    cols_write = cols_write if preserve_table_schema else list(df_write.columns)
                    on_exists = on_exists if preserve_table_schema else "replace"
                    query = f"delete from {table};" if preserve_table_schema else f"drop table {table};"
                    write_q = write_q if preserve_table_schema else True

                con.execute(query) if (query is not None) else None
                df_write[cols_write].to_sql(
                    table, con, if_exists = on_exists, index = None
                ) if write_q else None

            else:
                df_write.to_sql(table, con, if_exists = "replace", index = None)
