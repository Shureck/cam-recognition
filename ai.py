from mtcnn.mtcnn import MTCNN
import statistics
from keras_facenet import FaceNet
import os
import files
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import cv2
import numpy
import keras
import imutils
import inception_resnet_v1

# Создание сети нахождения лиц
detector = MTCNN()
embedder = FaceNet()
capture = None
directory_path = 'photos'

# Получить дистанцию лица
def get_distance(model, face):
    return embedder.embeddings([face])[0]

base = {}

def extract_face(path):
    frame = cv2.imread(path)

    if frame.shape[0] < frame.shape[1]:
        frame = imutils.resize(frame, height=1000)
    else:
        frame = imutils.resize(frame, width=1000)

    f_frame = frame
    image_size = numpy.asarray(frame.shape)[0:2]
    faces_boxes = detector.detect_faces(frame)
    image_detected = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    marked_color = (0, 255, 0, 1)

    # Работа с лицами
    if len(faces_boxes) > 1:
        return "Больше одного лица"
    else:
        for face_box in faces_boxes:
            x, y, w, h = face_box['box']

            # Выравнивание лица
            d = h - w  # Разница между высотой и шириной
            w = w + d  # Делаем изображение квадратным
            x = numpy.maximum(x - round(d / 2), 0)
            x1 = numpy.maximum(x, 0)
            y1 = numpy.maximum(y, 0)
            x2 = numpy.minimum(x + w, image_size[1])
            y2 = numpy.minimum(y + h, image_size[0])

            f_cropped = f_frame[y1:y2, x1:x2, :]

            f_face_image = cv2.resize(f_cropped, (160, 160), interpolation=cv2.INTER_AREA)
            print(len(f_face_image))
            if face_box['confidence'] > 0.90:  # 0.99 - уверенность сети в процентах что это лицо
                cv2.imwrite("ssss.jpg", f_face_image)
                # print(is_success, len(im_buf_arr))
                # return im_buf_arr.tobytes()
                return "Lox"
            else:
                return "Не удалось достоверно определить лицо"
                

def retrain_model(path, name):

    image = cv2.imread(path)
    # Замена BGR на RGB
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Получить размеры изображения
    image_size = numpy.asarray(image.shape)[0:2]
    # Получение списка лиц с координатами и значением уверенности
    faces_boxes = detector.detect_faces(image)
    # Работа с лицами
    if faces_boxes:

        # Координаты лица
        x, y, w, h = faces_boxes[0]['box']

        # Выравнивание лица
        d = h - w  # Разница между высотой и шириной
        w = w + d  # Делаем изображение квадратным
        x = numpy.maximum(x - round(d / 2), 0)
        x1 = numpy.maximum(x, 0)
        y1 = numpy.maximum(y, 0)
        x2 = numpy.minimum(x + w, image_size[1])
        y2 = numpy.minimum(y + h, image_size[0])

        # Получение картинки с лицом
        cropped = image[y1:y2, x1:x2, :]
        face_image = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_AREA)

        # Сохранение суммы евклидова пространства
        base[name].append(get_distance(embedder, image))

for dirname in files.get_folders(directory_path):

    base[dirname] = []
    for file in os.listdir(f'{directory_path}/' + dirname):

        if file.endswith('.jpg') or file.endswith('.JPG'):
            retrain_model(f'{directory_path}/' + dirname + '/' + file, dirname)

# frame_id = 0  # Инициализация счётчика кадров
# face_n = 0  # Инициализация счётчика лиц

# for file in os.listdir('C:/Users/sasha/OneDrive/Документы/Проекты Python/Фестиваль молодежи_ для участника'):

#     if file.endswith('.jpg') or file.endswith('.JPG'):
#         path = 'C:/Users/sasha/OneDrive/Документы/Проекты Python/Фестиваль молодежи_ для участника/'+file
#     else:
#         continue

#     frame_id += 1

#     success = True
#     frame = cv2.imread(path)

#     if success:
#         if frame.shape[0] < frame.shape[1]:
#             frame = imutils.resize(frame, height=1000)
#         else:
#             frame = imutils.resize(frame, width=1000)

#         f_frame = frame
#         image_size = numpy.asarray(frame.shape)[0:2]
#         faces_boxes = detector.detect_faces(frame)
#         image_detected = frame.copy()
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         marked_color = (0, 255, 0, 1)

#         # Работа с лицами
#         if faces_boxes:

#             for face_box in faces_boxes:
#                 face_n += 1
#                 x, y, w, h = face_box['box']

#                 # Выравнивание лица
#                 d = h - w  # Разница между высотой и шириной
#                 w = w + d  # Делаем изображение квадратным
#                 x = numpy.maximum(x - round(d / 2), 0)
#                 x1 = numpy.maximum(x, 0)
#                 y1 = numpy.maximum(y, 0)
#                 x2 = numpy.minimum(x + w, image_size[1])
#                 y2 = numpy.minimum(y + h, image_size[0])

#                 # Получение картинки с лицом
#                 cropped = frame[y1:y2, x1:x2, :]
#                 f_cropped = f_frame[y1:y2, x1:x2, :]

#                 face_image = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_AREA)
#                 f_face_image = cv2.resize(f_cropped, (160, 160), interpolation=cv2.INTER_AREA)
#                 distance = get_distance(embedder, face_image)
#                 x, y, w, h = face_box['box']
#                 d = h - w  # Разница между высотой и шириной
#                 w = w + d  # Делаем изображение квадратным
#                 x = numpy.maximum(x - round(d / 2), 0)
#                 x1 = numpy.maximum(x - round(w / 4), 0)
#                 y1 = numpy.maximum(y - round(h / 4), 0)
#                 x2 = numpy.minimum(x + w + round(w / 4), image_size[1])
#                 y2 = numpy.minimum(y + h + round(h / 4), image_size[0])
#                 if face_box['confidence'] > 0.97:  # 0.99 - уверенность сети в процентах что это лицо
#                     identity = None
#                     difference = None
#                     min_difference = 1
#                     median = None
#                     min_median = 1
#                     faces = {}
#                     for name, base_distances in base.items():
#                         faces[name] = []
#                         for base_distance in base_distances:
#                             difference = numpy.linalg.norm(base_distance - distance)
#                             if difference < min_difference:
#                                 print('difference - ' + str(difference))
#                                 faces[name].append(difference)

#                     if faces:
#                         for name, items in faces.items():
#                             # Идентификация только участвуют два и больше лиц
#                             if items and len(items) >= 2:
#                                 print(name)
#                                 print(items)
#                                 median = statistics.median(items)
#                                 if median < min_median:
#                                     print('median - ' + str(median))
#                                     min_median = median
#                                     identity = name

#                     if identity:
#                         cv2.putText(
#                             image_detected,
#                             identity,
#                             (x1 + 10, y2 + 20),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             1,
#                             (0, 255, 0),
#                             1
#                         )

#                         cv2.rectangle(
#                             image_detected,
#                             (x1, y1),
#                             (x2, y2),
#                             (0, 255, 0, 1),
#                             1
#                         )
#                         # Сохранение изображения лица на диск в директорию recognized
#                         if median < 0.95:
#                             cv2.imwrite(f'{directory_path}/{identity}/' + file+ '.jpg', f_face_image)
#                             retrain_model(f'{directory_path}/{identity}/' + file+ '.jpg', identity)

#                         user_folder = f"recognized/{identity}"
#                         os.makedirs(user_folder, exist_ok=True)
#                         cv2.imwrite(f'recognized/{identity}/' + str(median) + '.' + str(face_n)
#                                     + '.jpg', f_frame)

#                         # Информируем консоль
#                         print('\033[92m' + str(identity) + ' - ' + str(min_median) + '\033[0m')

#                     else:
#                         cv2.rectangle(
#                             image_detected,
#                             (x1, y1),
#                             (x2, y2),
#                             (255, 255, 255, 1),
#                             1
#                         )
#                         print('\033[91mNone\033[0m')

#                 else:
#                     print('\033[91mFalse\033[0m')

#         # Сохраняем кадр с видео
#         os.makedirs("demo/frames/", exist_ok=True)
#         cv2.imwrite('demo/frames/' + str(frame_id) + '.jpg', image_detected)
#         print('frame ' + str(frame_id))

#     else:
#         break
