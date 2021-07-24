var data = new FormData();

// Image Upload
const contentInpFile = document.getElementById("contentInputFile");
const styleInpFile = document.getElementById("styleInputFile");

// Image Upload Preview
const contentPreviewContainer = document.getElementById("contentImagePreview");
const stylePreviewContainer = document.getElementById("styleImagePreview");

const contentPreviewImage = contentPreviewContainer.querySelector(".preview_image");
const contentDefaultText = contentPreviewContainer.querySelector(".preview_default_text");

const stylePreviewImage = stylePreviewContainer.querySelector(".preview_image");
const styleDefaultText = stylePreviewContainer.querySelector(".preview_default_text");

function selectUploadFile(file, previewImage, previewDefaultText, dataFileKey){
    if(file){
        previewImage.style.display = "block";
        previewDefaultText.style.display = "none";

        const reader = new FileReader();
        reader.addEventListener("load", function(){
            previewImage.setAttribute("src", this.result);

            data.append(dataFileKey, file);
            console.log("File set for " + dataFileKey);
        });
        reader.readAsDataURL(file);
    }
    else{
        console.log("Failed to load image");
        previewDefaultText.style.display = null;
        previewImage.style.display = null;
        previewImage.setAttribute("src", "");
    }
}

// contentInpFile.addEventListener("change", function(){selectUploadFile(contentPreviewImage, contentDefaultText)});
// styleInpFile.addEventListener("change", function(){selectUploadFile(stylePreviewImage, styleDefaultText)});
contentInpFile.addEventListener("change", function(){
    selectUploadFile(this.files[0], contentPreviewImage, contentDefaultText, 'content_file');
});
styleInpFile.addEventListener("change", function(){
    selectUploadFile(this.files[0], stylePreviewImage, styleDefaultText, 'style_file');
});

// Upload images to backend
function uploadImages(){
    const postImages = async() => {
        const response = await fetch('http://127.0.0.1:5000/post_images',{
            method: 'POST',
            body : data,
            headers: {
                'credentials': "same-origin",
                'credentials': "include",
                'Origin': 'http://localhost:5500/'
            }
        })
        .then(function(response) {
            if (response.status !== 200){
                response.json().then(function(body){
                    console.log(`Status code: ${response.status}, Mrror message ${body["msg"]}`);
                });
            }
            else{
                response.json().then(function(body){
                    console.log(`Sucess! \nStatus code: ${response.status}, Message ${body["msg"]}`);
                });
            }
        });
    }
    postImages();
}

const imageUploadButton = document.getElementById("imageUploadButton");
imageUploadButton.addEventListener("click", uploadImages);