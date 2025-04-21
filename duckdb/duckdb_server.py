from fastapi import FastAPI
import duckdb
from fastapi.responses import JSONResponse

app = FastAPI()

# Connect to DuckDB
#conn = duckdb.connect('urban_sounds.duckdb', access_mode='READ_ONLY')
conn = duckdb.connect('urban_sounds.duckdb', read_only=True)

@app.get("/data")
async def get_data():
    query = "SELECT * FROM your_table"
    data = conn.execute(query).fetchall()
    return JSONResponse(content=data)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
