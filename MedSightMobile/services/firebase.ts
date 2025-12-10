import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

// Firebase configuration - MedSightMobile project
// Android app sẽ tự động dùng google-services.json
const firebaseConfig = {
  apiKey: "AIzaSyAmn7xk4mWdq2I-N5umQk94CjcnS5fZNto",
  authDomain: "medsightmobile.firebaseapp.com",
  projectId: "medsightmobile",
  storageBucket: "medsightmobile.firebasestorage.app",
  messagingSenderId: "1044083966032",
  appId: "1:1044083966032:android:758996eb9b68b7ec729155"
};

// Initialize Firebase
export const app = initializeApp(firebaseConfig);

// Initialize Firebase Auth
export const auth = getAuth(app);

// Initialize Firestore
export const db = getFirestore(app);

export default app;
