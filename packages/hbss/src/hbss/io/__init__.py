"""Ingestion for NISAR/Sentinel-1 (Dolphin, ISCE3), GNSS (.pos), and RAiDER ZTD fields.

All internal representations are xarray DataArrays with CF-compliant metadata,
chunked to spatial tiles on disk (Zarr).
"""
