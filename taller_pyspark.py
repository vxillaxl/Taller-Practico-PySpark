import os
import shutil
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    avg,
    col,
    count,
    countDistinct,
    date_format,
    dayofmonth,
    hour,
    lit,
    max as spark_max,
    min as spark_min,
    month,
    rank,
    regexp_replace,
    round as spark_round,
    row_number,
    sum as spark_sum,
    to_timestamp,
    trim,
    when,
    year,
)

COLUMNAS_15 = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
    "TotalAmount",
    "Year",
    "Month",
    "Day",
    "Hour",
    "DayOfWeek",
    "InvoiceType",
]

ROOT = Path(__file__).resolve().parent
DATOS = ROOT / "data" / "raw" / "online_retail.csv"
RESULTADOS = ROOT / "resultados"
URL_DATASET = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def preparar_windows():
    if os.name != "nt":
        return
    hadoop_home = ROOT / "tools" / "hadoop"
    bin_dir = hadoop_home / "bin"
    winutils = bin_dir / "winutils.exe"
    hadoop_dll = bin_dir / "hadoop.dll"
    if not winutils.exists() or not hadoop_dll.exists():
        bin_dir.mkdir(parents=True, exist_ok=True)
        import urllib.request

        base = "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.5/bin"
        urllib.request.urlretrieve(f"{base}/winutils.exe", winutils)
        urllib.request.urlretrieve(f"{base}/hadoop.dll", hadoop_dll)
    os.environ["HADOOP_HOME"] = str(hadoop_home)
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def crear_spark():
    preparar_windows()
    os.environ.setdefault("JAVA_HOME", r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    return (
        SparkSession.builder.appName("Taller1_ETL_OnlineRetail")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def limpiar_datos(df):
    filas_antes = df.count()
    df = (
        df.withColumn("InvoiceNo", trim(col("InvoiceNo").cast("string")))
        .withColumn("StockCode", trim(col("StockCode").cast("string")))
        .withColumn("Description", regexp_replace(trim(col("Description")), r"\s+", " "))
        .withColumn("Country", trim(col("Country")))
        .withColumn("CustomerID", col("CustomerID").cast("int"))
    )
    df = df.dropDuplicates()
    df = df.filter(
        col("InvoiceNo").isNotNull()
        & (col("InvoiceNo") != "")
        & col("StockCode").isNotNull()
        & (col("StockCode") != "")
        & col("Description").isNotNull()
        & (col("Description") != "")
        & col("InvoiceDate").isNotNull()
        & (col("UnitPrice") >= 0)
        & col("Quantity").isNotNull()
        & col("Country").isNotNull()
        & (col("Country") != "")
    )
    filas_despues = df.count()
    print("filas antes:", filas_antes)
    print("filas despues:", filas_despues)
    return df


def extraer_dataset():
    DATOS.parent.mkdir(parents=True, exist_ok=True)
    if DATOS.exists() and DATOS.stat().st_size > 0:
        print("dataset ya esta descargado")
        return
    print("descargando dataset de UCI...")
    respuesta = requests.get(URL_DATASET, timeout=120)
    respuesta.raise_for_status()
    with zipfile.ZipFile(BytesIO(respuesta.content)) as zf:
        xlsx = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
        with zf.open(xlsx) as fh:
            crudo = pd.read_excel(fh, engine="openpyxl")
    crudo.to_csv(DATOS, index=False, encoding="utf-8")
    print("listo,", len(crudo), "filas")


def guardar_csv(df, nombre):
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = RESULTADOS / nombre
    tmp = RESULTADOS / f"_tmp_{destino.stem}"
    if tmp.exists():
        shutil.rmtree(tmp)
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(tmp.as_posix())
    part = next(tmp.glob("part-*.csv"))
    if destino.exists():
        destino.unlink()
    shutil.move(str(part), str(destino))
    shutil.rmtree(tmp)
    print("  ->", destino.name)
    df.show(truncate=False)


def pregunta_01_total_facturas(df):
    print("\nPregunta 1: número total de facturas")
    resultado = df.agg(countDistinct("InvoiceNo").alias("total_facturas"))
    guardar_csv(resultado, "01_total_facturas.csv")


def pregunta_02_clientes_unicos(df):
    print("\nPregunta 2: número de clientes únicos")
    resultado = df.filter(col("CustomerID").isNotNull()).agg(
        countDistinct("CustomerID").alias("clientes_unicos")
    )
    guardar_csv(resultado, "02_clientes_unicos.csv")


def pregunta_03_ingreso_total(df):
    print("\nPregunta 3: ingreso total (Quantity * UnitPrice)")
    resultado = df.agg(spark_round(spark_sum("TotalAmount"), 2).alias("ingreso_total"))
    guardar_csv(resultado, "03_ingreso_total.csv")


def pregunta_04_producto_mas_vendido(df):
    print("\nPregunta 4: producto más vendido en cantidad")
    ventana = Window.orderBy(col("cantidad_total").desc())
    resultado = (
        df.groupBy("StockCode", "Description")
        .agg(spark_sum("Quantity").alias("cantidad_total"))
        .withColumn("row_number", row_number().over(ventana))
        .filter(col("row_number") == 1)
        .drop("row_number")
    )
    guardar_csv(resultado, "04_producto_mas_vendido.csv")


def pregunta_05_cliente_mayor_compra(df):
    print("\nPregunta 5: cliente con mayor volumen de compra")
    df_clientes = (
        df.select("CustomerID", "Country")
        .filter(col("CustomerID").isNotNull())
        .dropDuplicates(["CustomerID"])
    )
    df_compras = (
        df.filter(col("CustomerID").isNotNull())
        .groupBy("CustomerID")
        .agg(spark_round(spark_sum("TotalAmount"), 2).alias("volumen_compra"))
    )
    ventana = Window.orderBy(col("volumen_compra").desc())
    resultado = (
        df_compras.join(df_clientes, "CustomerID", "inner")
        .withColumn("rank", rank().over(ventana))
        .filter(col("rank") == 1)
        .drop("rank")
    )
    guardar_csv(resultado, "05_cliente_mayor_compra.csv")


def pregunta_06_top5_paises_fuera_uk(df):
    print("\nPregunta 6: 5 países que más compran fuera de Reino Unido")
    resultado = (
        df.filter(col("Country") != "United Kingdom")
        .groupBy("Country")
        .agg(spark_round(spark_sum("TotalAmount"), 2).alias("ingreso_total"))
        .orderBy(col("ingreso_total").desc())
        .limit(5)
    )
    guardar_csv(resultado, "06_top5_paises_fuera_uk.csv")


def pregunta_07_ticket_promedio(df):
    print("\nPregunta 7: ticket promedio por factura")
    resultado = (
        df.groupBy("InvoiceNo")
        .agg(spark_sum("TotalAmount").alias("total_factura"))
        .agg(spark_round(avg("total_factura"), 2).alias("ticket_promedio"))
    )
    guardar_csv(resultado, "07_ticket_promedio.csv")


def pregunta_08_productos_por_factura(df):
    print("\nPregunta 8: mínimo, máximo y promedio de productos por factura")
    resultado = (
        df.groupBy("InvoiceNo")
        .agg(count("*").alias("productos"))
        .agg(
            spark_min("productos").alias("min_productos"),
            spark_max("productos").alias("max_productos"),
            spark_round(avg("productos"), 2).alias("promedio_productos"),
        )
    )
    guardar_csv(resultado, "08_productos_por_factura.csv")


def pregunta_09_mes_mas_ventas(df):
    print("\nPregunta 9: mes del año con más ventas")
    resultado = (
        df.groupBy("Year", "Month")
        .agg(spark_round(spark_sum("TotalAmount"), 2).alias("ingreso_total"))
        .orderBy(col("ingreso_total").desc())
        .limit(1)
    )
    guardar_csv(resultado, "09_mes_mas_ventas.csv")


def pregunta_10_porcentaje_devoluciones(df):
    print("\nPregunta 10: porcentaje de facturas con devoluciones")
    facturas = df.select("InvoiceNo").distinct()
    devoluciones = (
        df.filter(col("Quantity") < 0)
        .select("InvoiceNo")
        .distinct()
        .withColumn("es_devolucion", lit(1))
    )
    resultado = (
        facturas.join(devoluciones, "InvoiceNo", "left")
        .agg(
            countDistinct("InvoiceNo").alias("total_facturas"),
            spark_sum("es_devolucion").alias("facturas_con_devolucion"),
        )
        .withColumn(
            "porcentaje_devoluciones",
            spark_round(col("facturas_con_devolucion") / col("total_facturas") * 100, 2),
        )
    )
    guardar_csv(resultado, "10_porcentaje_devoluciones.csv")


def main():
    extraer_dataset()

    spark = crear_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("leyendo csv")
    df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(DATOS.as_posix())
    )

    df = df.select(
        col("InvoiceNo"),
        col("StockCode"),
        col("Description"),
        col("Quantity"),
        to_timestamp(col("InvoiceDate"), "M/d/yyyy H:mm").alias("InvoiceDate"),
        col("UnitPrice"),
        col("CustomerID"),
        col("Country"),
    )

    print("limpiando datos")
    df = limpiar_datos(df)

    df = (
        df.withColumn("TotalAmount", col("Quantity") * col("UnitPrice"))
        .withColumn("Year", year("InvoiceDate"))
        .withColumn("Month", month("InvoiceDate"))
        .withColumn("Day", dayofmonth("InvoiceDate"))
        .withColumn("Hour", hour("InvoiceDate"))
        .withColumn("DayOfWeek", date_format("InvoiceDate", "EEEE"))
        .withColumn(
            "InvoiceType",
            when(col("Quantity") < 0, lit("Return")).otherwise(lit("Sale")),
        )
        .select(*COLUMNAS_15)
    )
    df.cache()
    print("filas:", df.count())
    print("columnas:", df.columns)

    pregunta_01_total_facturas(df)
    pregunta_02_clientes_unicos(df)
    pregunta_03_ingreso_total(df)
    pregunta_04_producto_mas_vendido(df)
    pregunta_05_cliente_mayor_compra(df)
    pregunta_06_top5_paises_fuera_uk(df)
    pregunta_07_ticket_promedio(df)
    pregunta_08_productos_por_factura(df)
    pregunta_09_mes_mas_ventas(df)
    pregunta_10_porcentaje_devoluciones(df)

    spark.stop()
    print("listo")


if __name__ == "__main__":
    main()
