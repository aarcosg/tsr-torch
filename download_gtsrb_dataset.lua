local pl = (require 'pl.import_into')()

local dataset = {}

dataset.path_remote_train = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip"
dataset.path_remote_test = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip"
dataset.path_remote_test_gt = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip"

dataset.train_folder = "train"
dataset.val_folder = "val"

function dataset.download_dataset()
   if not pl.path.isdir('GTSRB') then
    local zip_train = paths.basename(dataset.path_remote_train)
    local zip_test = paths.basename(dataset.path_remote_test)
    local zip_test_gt = paths.basename(dataset.path_remote_test_gt)

    print('Downloading dataset...')
    os.execute('wget ' .. dataset.path_remote_train .. '; ' ..
               'unzip ' .. zip_train .. '; '..
               'rm ' .. zip_train .. '; ' ..
               'mv GTSRB/Final_Training/Images/ GTSRB/train;' ..
               'rm -r GTSRB/Final_Training')
    os.execute('wget ' .. dataset.path_remote_test .. '; ' ..
               'unzip ' .. zip_test .. '; '..
               'rm ' .. zip_test .. '; ' ..
               'mkdir GTSRB/val; ' ..
               [[find GTSRB/Final_Test/Images/ -maxdepth 1 -name '*.ppm' -exec sh -c 'mv "$@" "$0"' GTSRB/val/ {} +;]] ..
               'rm -r GTSRB/Final_Test')
    os.execute('wget ' .. dataset.path_remote_test_gt .. '; ' ..
               'unzip ' .. zip_test_gt .. '; '..
               'rm ' .. zip_test_gt .. '; '..
               'mv GT-final_test.csv GTSRB/GT-final_test.csv')
  end
end

function dataset.move_val_images()
    print("Moving validation images to class folders")
    local val_dir = pl.path.join("GTSRB", dataset.val_folder)
    local csv_file_path = pl.path.join("GTSRB", "GT-final_test.csv")
    local csv_data = pl.data.read(csv_file_path)
    local filename_index = csv_data.fieldnames:index("Filename")
    local class_id_index = csv_data.fieldnames:index("ClassId")

    for _, image_metadata in ipairs(csv_data) do
        local image_name = image_metadata[filename_index]
        local image_path = pl.path.join(val_dir, image_name)
        local image_label = image_metadata[class_id_index]
        local class_folder_name = string.format("%05d", image_label)
        local class_folder_path = pl.path.join(val_dir, class_folder_name)
        if not pl.path.exists(class_folder_path) then
            pl.path.mkdir(class_folder_path)
        end
        pl.file.move(image_path, pl.path.join(class_folder_path, image_name))
    end
end


dataset.download_dataset()
dataset.move_val_images()



