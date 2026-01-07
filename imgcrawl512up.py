import sys
from PIL import Image, ExifTags
import os
from icrawler.builtin import GoogleImageCrawler

# Get the command-line arguments for the keywords and max_num
if len(sys.argv) != 3:
    print("Usage: python script.py <comma-separated-keywords> <max_num>")
    sys.exit(1)

keywords = sys.argv[1].split(',')
max_num = int(sys.argv[2])

# Process each keyword separately
for keyword in keywords:
    keyword = keyword.strip()  # Remove leading/trailing whitespace

    # Create the folder with the keyword name if it doesn't exist
    os.makedirs(keyword, exist_ok=True)

    # Create the GoogleImageCrawler with storage root_dir as the keyword
    google_crawler = GoogleImageCrawler(storage={"root_dir": keyword})

    # Crawl images with the given keyword and max_num
    google_crawler.crawl(keyword=keyword, max_num=max_num)

    print(f"Images of '{keyword}' are downloaded and saved in the '{keyword}' directory.")

    # Update the EXIF metadata to add the custom tag
    for image_file in os.listdir(keyword):
        image_path = os.path.join(keyword, image_file)

        try:
            cropped_image = Image.open(image_path)
            
            # Check if image dimensions are at least 512x512
            if cropped_image.size[0] < 512 or cropped_image.size[1] < 512:
                print(f"Ignored {image_file} due to dimensions below 512x512.")
                continue

            # Check if file size is over 1MB
            if os.path.getsize(image_path) > (1024 * 1024):
                print(f"Ignored {image_file} due to file size over 1MB.")
                continue

            exif_data = cropped_image._getexif()
            if exif_data is not None:
                exif_data = dict(exif_data)
                exif_data[ExifTags.TAGS["Software"]] = keyword.encode("utf-8")
                cropped_image.save(image_path, exif=exif_data)

                print(f"Added tag '{keyword}' to {image_file}.")
            else:
                print(f"Could not find EXIF metadata in {image_file}.")
        except Exception as e:
            print(f"An error occurred while adding the tag to {image_file}: {e}")

print("Tagging completed for all images.")
