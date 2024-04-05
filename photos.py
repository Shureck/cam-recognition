import os

async def save_user_photo(email: str, photo):
    # Создание папки пользователя, если ее нет
    user_folder = f"photos/{email}"
    os.makedirs(user_folder, exist_ok=True)

    # Проверка наличия уже загруженных фотографий
    existing_photos = os.listdir(user_folder)
    if len(existing_photos) >= 4:
        return {'status_code':400, 'detail':"Пользователь уже загрузил максимальное количество фотографий"}

    # Сохранение файла
    file_location = os.path.join(user_folder, photo.filename)
    with open(file_location, "wb") as file:
        file.write(photo.file.read())
    return file_location