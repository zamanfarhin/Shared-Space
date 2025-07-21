import React, { useState } from 'react';

const names = [
  { label: 'Satyajit Ray', value: 'satyajit_ray' },
  { label: 'Hayao Miyazaki', value: 'hayao_miyazaki' },
  { label: 'Orhan Pamuk', value: 'orhan_pamuk' },
  { label: 'Humayun Ahmed', value: 'humayun_ahmed' }
];
const API_URL = 'https://shared-space-backend.onrender.com/generate';
function SharedSpaceGenerator() {
  const [selected, setSelected] = useState('');
  const [profile, setProfile] = useState(null);

  const handleChange = async (e) => {
    const val = e.target.value;
    setSelected(val);

    const res = await fetch(`${API_URL}?name=${val}`);
    const data = await res.json();
    setProfile(data);
  };

  return (
    <div style={{ fontFamily: 'sans-serif', padding: '2rem', textAlign: 'center' }}>
      <h2>✨ A space shared by...</h2>
      <select onChange={handleChange} value={selected}>
        <option value="">Select a figure</option>
        {names.map((n) => (
          <option key={n.value} value={n.value}>{n.label}</option>
        ))}
      </select>

      {profile && (
        <div style={{ marginTop: '2rem' }}>
          <h3>{profile.name}</h3>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginTop: '1rem' }}>
            {profile.color_palette.map((hex, idx) => (
              <div key={idx} style={{
                backgroundColor: hex,
                width: '50px',
                height: '50px',
                borderRadius: '8px',
                border: '1px solid #ccc'
              }} title={hex}></div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default SharedSpaceGenerator;
