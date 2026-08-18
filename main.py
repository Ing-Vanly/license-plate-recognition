import cv2
from plate_finder import PlateFinder

finder = PlateFinder(
    min_plate_area=4100,
    max_plate_area=30000
)

image = cv2.imread("images/car.jpg")

if image is None:
    print("Could not load images/car.jpg")
    exit()

plates = finder.find_possible_plates(image)

print("Possible plates found:", len(plates))

for i, plate in enumerate(plates):
    cv2.imshow(f"Plate {i + 1}", plate)

    x, y, w, h = finder.coordinates[i]

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

cv2.imshow("Car Image", image)

print("Press Q to close the windows")

while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()