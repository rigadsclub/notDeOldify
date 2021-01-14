curl -X POST "http://127.0.0.1:5000/process" \
	-H "accept: image/jpg" \
	-H "Content-Type: application/json" \
	-d "{\"url\":\"http://www.digital-photo-secrets.com/image/ebooks/blackwhite/bw-landscape-alice.jpg\", \"render_factor\":35}" \
	--output colorized_image.png
