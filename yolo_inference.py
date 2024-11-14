from ultralytics import YOLO

model = YOLO('models/best.pt') # run line in 'football_training_yolo_v5.ipynb' in colab to use gpu, get best/last weights -> models 

results = model.predict('input_vids/08fd33_4.mp4', save=True)
print(results[0])
print('******************')
for box in results[0].boxes:
    print(box)



