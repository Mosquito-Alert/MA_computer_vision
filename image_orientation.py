import piexif

def get_exif_data(image_path):
    try:
        # Load the image's EXIF data
        exif_data = piexif.load(image_path)

        if exif_data:
            # Loop through each EXIF section
            for ifd in exif_data:
                print(f"--- {ifd} ---")
                
                # Ensure the section contains data before attempting to iterate
                if exif_data[ifd]:
                    for tag in exif_data[ifd]:
                        tag_name = piexif.TAGS[ifd].get(tag, {}).get("name", tag)
                        print(f"{tag_name}: {exif_data[ifd][tag]}")
                else:
                    print(f"No data in {ifd} section.")
        else:
            print("No EXIF data found.")
    except Exception as e:
        print(f"Error: {e}")


image_path = 'WhatsApp Image 2024-09-09 at 19.49.54.jpeg'
get_exif_data(image_path)
