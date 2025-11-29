import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyD5t4hOFqy1EAw7VOcZXOJSY9ovgjERxHE",
  authDomain: "wedai-dcecf.firebaseapp.com",
  projectId: "wedai-dcecf",
  storageBucket: "wedai-dcecf.firebasestorage.app",
  messagingSenderId: "795123587676",
  appId: "1:795123587676:web:4bf4ee979eafd6b57d2499",
  measurementId: "G-XD6FBE6KZD"
};

// Initialize Firebase
export const app = initializeApp(firebaseConfig);

// Initialize Firebase Auth
export const auth = getAuth(app);

// Initialize Firestore
export const db = getFirestore(app);

export default app;
