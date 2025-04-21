import duckdb
import json

# Connect to the DuckDB database
con = duckdb.connect('urban_sounds.duckdb')

# Execute the SQL query to retrieve data
query = "SELECT * FROM mqtt_messages"
result = con.execute(query).fetchdf()

# Print the retrieved data
print(result)

# If you want to access the payload_fields as a dictionary:
result['payload_fields'] = result['payload_fields'].apply(json.loads)
print(result)

con.close()