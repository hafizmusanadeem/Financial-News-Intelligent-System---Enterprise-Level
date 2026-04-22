from src.fni.core.constants import db_name, user, password
import psycopg2

conn = psycopg2.connect(
    db_name = db_name,
    user = user,
    password = password
)
conn.close()