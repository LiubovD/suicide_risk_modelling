import qrcode

url = "https://liubovd.github.io/suicide_mapping_Rhode_Island/"
img = qrcode.make(url)
img.save("dashboard_qr.png")