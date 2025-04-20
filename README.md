# Smart Album

A playful online photo album that lets you share your photos with friends and enhance the fun by adding personalized background music to each picture.

## Key Features
![image](https://github.com/user-attachments/assets/edc6b4ec-bdbb-4ac2-8fc9-47463b4acfab)

![image](https://github.com/user-attachments/assets/7cd6ac0e-a27f-4db5-b7f8-9421dde999b1)

![image](https://github.com/user-attachments/assets/cbc14243-d894-44ec-8ab8-4917dffdf3a7)


- **Semantic Search with Embedding Optimization**: Photos uploaded to the application are optimized using embedding techniques, enabling intelligent semantic search capabilities. This allows users to search for photos not just by keywords, but by the underlying semantic meaning.
  
- **Direct Image-to-Music Generation**: The application supports direct generation of background music from images, reflecting the mood and content of the visual input. This feature provides a seamless experience for users to create personalized music tracks from their photos.

## Features

- **Photo Upload**: Users can upload photos to the application.
- **Image Analysis**: The application analyzes the uploaded images to extract descriptions, keywords, emotions, and actions.
- **Music Generation**: Based on the analysis, the application generates a music track that reflects the mood and content of the image.
- **Audio Playback**: Automatically plays the generated music and allows users to download it in MP3 format.
- **Search Functionality**: Users can search for photos by keywords, emotions, or semantic meaning.
- **Responsive Design**: User-friendly interface with a responsive design for various devices.

## Getting Started

These instructions will help you set up a local development environment to run the project on your machine.

### Prerequisites

- Python 3.x
- Flask
- Other dependencies listed in `requirements.txt`

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/smart-photo-music-generator.git
   cd smart-photo-music-generator
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source venv/bin/activate
     ```

4. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables:**

   Create a `.env` file in the root directory and add your API keys:

   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   STABLE_AUDIO_API_KEY=your_stable_audio_api_key
   ```

6. **Initialize the database:**

   The database will be initialized automatically when you first run the application.

### Running the Application

1. **Start the Flask development server:**

   ```bash
   flask run
   ```

2. **Open your web browser and visit:**

   ```
   http://127.0.0.1:5000/
   ```

3. **Upload a photo and generate music:**
   - Use the interface to upload a photo.
   - Experience searching for photos by atmosphere, emotion, or action.
   ![image](https://github.com/user-attachments/assets/ed09caf7-52f6-43eb-b419-10fc17453073)
   ![image](https://github.com/user-attachments/assets/5cbab5c4-08c2-475b-97e7-cef294cd0c76)

   - Click the "Generate Music" button to create a music track.
   - Listen to the generated music and download it if desired.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors and open-source libraries used in this project.

## Copyright

© 2025 @ImportHe (zhonghe0615). All rights reserved. 
