from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List  # Untuk mendukung daftar input
import numpy as np
import pickle

# Load model dan scaler
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    raise Exception(f"Error loading scaler: {str(e)}")

# Inisialisasi FastAPI
app = FastAPI(
    title="Clustering Pendidikan API (Local)",
    description="API untuk prediksi cluster pendidikan SD Indonesia secara lokal.",
    version="1.0.0"
)

# Skema input data
class InputData(BaseModel):
    Provinsi: int
    Mengulang: int
    Rombongan_Belajar: int
    Ruang_kelas_baik: int
    Ruang_kelas_rusak: int
    Rasio_Putus_Sekolah_per_Siswa: float
    Rasio_Guru_S1_per_Siswa: float

# Endpoint untuk prediksi
@app.post("/predict")
async def predict(data: List[InputData]):  # Ubah parameter menjadi List[InputData]
    try:
        # Validasi minimal 2 sampel
        if len(data) < 2:
            raise HTTPException(status_code=400, detail="At least 2 samples are required for AgglomerativeClustering")

        # Convert input ke array
        input_array = np.array([[
            d.Provinsi,
            d.Mengulang,
            d.Rombongan_Belajar,
            d.Ruang_kelas_baik,
            d.Ruang_kelas_rusak,
            d.Rasio_Putus_Sekolah_per_Siswa,
            d.Rasio_Guru_S1_per_Siswa
        ] for d in data])

        # Preprocessing (scaling)
        input_scaled = scaler.transform(input_array)

        # Gunakan fit_predict untuk AgglomerativeClustering
        predictions = model.fit_predict(input_scaled)

        # Output sebagai daftar hasil prediksi
        return [{"Cluster": int(pred)} for pred in predictions]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

