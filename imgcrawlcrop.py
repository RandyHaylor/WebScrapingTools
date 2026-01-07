import sys
from PIL import Image, ExifTags
import os
from icrawler.builtin import GoogleImageCrawler

# Get the command-line arguments for the keyword and max_num
if len(sys.argv) != 3:
print("Usage: python script.py <keyword> <max_num>")
sys.exit(1)

keyword = sys.argv[1]
max_num = int(sys.argv[2])

# Create the folder with the keyword name if it doesn't exist
os.makedirs(keyword, exist_ok=True)

# Create the GoogleImageCrawler with storage root_dir as the keyword
google_crawler = GoogleImageCrawler(storage={"root_dir": keyword})

# Crawl images with the given keyword and max_num
google_crawler.crawl(keyword=keyword, max_num=max_num)

print(f"Images of '{keyword}' are downloaded and saved in the '{keyword}' directory.")

# Get a list of all image files in the current working directory
image_files = [f for f in os.listdir(keyword) if f.endswith((".jpg", ".jpeg", ".png"))]

# Iterate over each image file
for image_file in image_files:
# Get the full path of the image
image_path = os.path.join(keyword, image_file)

# Check the file size
file_size = os.path.getsize(image_path)

# Ignore files over 1MB in size
if file_size <= 1024 * 1024:
# Open the image
image = Image.open(image_path)

# Get the original dimensions
width, height = image.size

# Determine the size of the square crop
size = min(width, height)

# Calculate the crop boundaries
left = (width - size) // 2
top = (height - size) // 2
right = left + size
bottom = top + size

# Crop the image to a square
cropped_image = image.crop((left, top, right, bottom))

# Save the cropped image in the folder with the keyword name and the same filename
cropped_image.save(os.path.join(keyword, image_file))

print(f"Cropped {image_file} to a 1:1 ratio and saved in the '{keyword}' folder.")

# Update the EXIF metadata to add the custom tag
try:
exif_data = cropped_image._getexif()
if exif_data is not None:
exif_data = dict(exif_data)
exif_data[ExifTags.TAGS["Software"]] = keyword.encode("utf-8")
cropped_image.save(os.path.join(keyword, image_file), exif=exif_data)

print(f"Added tag '{keyword}' to {image_file}.")
else:
print(f"Could not find EXIF metadata in {image_file}.")
except Exception as e:
print(f"An error occurred while adding the tag to {image_file}: {e}")
else:
print(f"Ignored {image_file} due to file size over 1MB.")

print("Crop and tag completed for all images.")