from ultralytics import YOLO

# Modelo base
modelo = YOLO("modelo_ia/yolov8n.pt")

# Entrenamiento
resultado = modelo.train(
    data="dataset_filtrado/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="supervision_examenes",
    project="runs"
)

print("====================================")
print("ENTRENAMIENTO FINALIZADO")
print("====================================")
print("Los resultados están en:")
print("runs/supervision_examenes")