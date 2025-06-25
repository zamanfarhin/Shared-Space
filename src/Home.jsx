import { useState } from 'react';

function Home() {
  const [name, setName] = useState('');
  const [inspiration, setInspiration] = useState('');
  const [pinterest, setPinterest] = useState('');
  const [roomType, setRoomType] = useState('');
  const [image, setImage] = useState(null);
  const [story, setStory] = useState('');

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(URL.createObjectURL(file));
    }
  };

  const handleGenerateStory = () => {
    const poeticLine = `Sitting in your ${roomType}, inspired by ${inspiration}, the air feels soft and textured—like a quiet frame in a ${inspiration} film.`;
    setStory(`Hello ${name}, here’s your space:\n\n${poeticLine}`);
  };

  return (
    <div className="min-h-screen bg-white text-gray-800 p-6 max-w-xl mx-auto">
      <h1 className="text-3xl font-semibold mb-6">Design Your Space</h1>

      <div className="space-y-4">
        <input
          type="text"
          placeholder="Your name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full p-2 border border-gray-300 rounded"
        />

        <input
          type="text"
          placeholder="Who inspires you? (e.g. Satyajit Ray, Pamuk, Miyazaki)"
          value={inspiration}
          onChange={(e) => setInspiration(e.target.value)}
          className="w-full p-2 border border-gray-300 rounded"
        />

        <input
          type="text"
          placeholder="Pinterest board (optional)"
          value={pinterest}
          onChange={(e) => setPinterest(e.target.value)}
          className="w-full p-2 border border-gray-300 rounded"
        />

        <input
          type="text"
          placeholder="What space are you designing? (e.g. bedroom, studio)"
          value={roomType}
          onChange={(e) => setRoomType(e.target.value)}
          className="w-full p-2 border border-gray-300 rounded"
        />

        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          className="w-full p-2"
        />
      </div>

      <div className="mt-6">
        <h2 className="text-xl font-medium mb-2">Preview</h2>
        <p><strong>Name:</strong> {name}</p>
        <p><strong>Inspiration:</strong> {inspiration}</p>
        <p><strong>Pinterest:</strong> {pinterest}</p>
        <p><strong>Room Type:</strong> {roomType}</p>
        {image && <img src={image} alt="Uploaded" className="mt-4 w-64 rounded shadow" />}
      </div>

      <button
        onClick={handleGenerateStory}
        className="mt-6 px-4 py-2 bg-black text-white rounded hover:bg-gray-800"
      >
        Generate Story
      </button>

      {story && (
        <div className="mt-6 p-4 border border-gray-200 rounded bg-gray-50 whitespace-pre-wrap">
          {story}
        </div>
      )}
    </div>
  );
}

export default Home;
