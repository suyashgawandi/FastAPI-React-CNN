import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; 

function App() {

  const [selectedImage, setSelectedImage] = useState(null);
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleFileChange = (event) =>{
    setSelectedImage(event.target.files[0])
  };

  const handleSubmit = async(event)=>{
    event.preventDefault(); //prevents default form behaviour 
    if(selectedImage){
      const image= new FormData();
      image.append("file", selectedImage);
      setLoading(true);
      setPrediction(null);

      try{
        const response = await axios.post(
          "http://localhost:8000/predict/image",
          image,
          {headers: {
            "Content-Type" : "multipart/form-data"
          },
        });
        setPrediction(response.data.prediction);
      }
      catch(error){
        console.error("Error uploading image: ", error);
      }
      finally{
        setLoading(false);
      }
    }
  };
  
  return(
    <div className='App'>
      <h1>MNIST Digit Prediction</h1>
      
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange}/>
        <button type="submit">Predict Image</button>
      </form>
      {loading && <p>Loading....</p>}
      {prediction!==null &&(
        <div>
          <h2>Predicted Image: {prediction}</h2>
        </div>
      )}
    </div>

  );
}

export default App;
