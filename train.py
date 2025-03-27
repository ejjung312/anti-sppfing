from ultralytics import YOLO

model = YOLO()

def main():
    model.train(data='dataset/splitdata/dataOffline.yaml', epochs=3)

if __name__ == '__main__':
    main()