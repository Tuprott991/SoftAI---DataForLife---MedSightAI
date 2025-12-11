import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { auth, db, app } from '@/services/firebase';
import { signOut, onAuthStateChanged, PhoneAuthProvider, signInWithCredential } from 'firebase/auth';
import { doc, getDoc, setDoc, collection } from 'firebase/firestore';
import { getVietnameseAuthError, normalizePhoneNumber } from '@/utils/otp-utils';

// Mock storage for users (simulates Firestore for offline/testing)
const mockUserDatabase: Record<string, any> = {};

// Test phone numbers and OTP configuration
const TEST_PHONES = ['+16505551234', '0999999999'];
const TEST_OTP = '123456';

export interface AuthUser {
  uid: string;
  phoneNumber: string | null;
  email: string | null;
  displayName: string | null;
}

export interface UserProfile {
  uid: string;
  phoneNumber: string;
  name?: string;
  age?: number;
  address?: string;
  createdAt: string;
  updatedAt?: string;
}

interface AuthContextType {
  user: AuthUser | null;
  userProfile: UserProfile | null;
  isLoading: boolean;
  isSignedIn: boolean;
  isNewUser: boolean;
  phoneNumber: string;
  setPhoneNumber: (phone: string) => void;
  verificationId: string | null;
  setVerificationId: (id: string | null) => void;
  sendOTP: (phoneNumber: string, appVerifier: any) => Promise<string>;
  verifyOTP: (otp: string) => Promise<any>;
  signOutUser: () => Promise<void>;
  updateUserProfile: (profile: Partial<UserProfile>) => Promise<void>;
  error: string | null;
  setError: (error: string | null) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isNewUser, setIsNewUser] = useState(false);
  const [phoneNumber, setPhoneNumber] = useState('');
  const [verificationId, setVerificationId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Check if user exists in Firestore (with mock fallback)
  const checkUserExists = async (uid: string): Promise<boolean> => {
    try {
      console.log('Auth: Checking if user exists, UID:', uid);
      
      try {
        // Try to get from Firestore
        const userDocRef = doc(db, 'users', uid);
        const userDoc = await getDoc(userDocRef);
        
        if (userDoc.exists()) {
          console.log('Auth: User found in Firestore');
          setUserProfile(userDoc.data() as UserProfile);
          setIsNewUser(false);
          return true;
        }
      } catch (firestoreErr) {
        console.log('Auth: Firestore unavailable, checking mock database');
      }
      
      // Fallback to mock database
      if (mockUserDatabase[uid]) {
        console.log('Auth: User found in mock database');
        setUserProfile(mockUserDatabase[uid]);
        setIsNewUser(false);
        return true;
      }
      
      console.log('Auth: User NOT found - new user');
      setIsNewUser(true);
      return false;
    } catch (err) {
      console.error('Auth: Error checking user:', err);
      return false;
    }
  };

  useEffect(() => {
    const initializeAuthListener = async () => {
      // Listen to auth state changes
      const unsubscribe = onAuthStateChanged(auth, async (currentUser: any) => {
        console.log('Auth state changed:', currentUser ? `User: ${currentUser.uid}` : 'No user');
        
        if (currentUser) {
          const authUser: AuthUser = {
            uid: currentUser.uid,
            phoneNumber: currentUser.phoneNumber,
            email: currentUser.email,
            displayName: currentUser.displayName,
          };
          setUser(authUser);
          
          // Check if user exists in Firestore
          await checkUserExists(currentUser.uid);
        } else {
          setUser(null);
          setUserProfile(null);
          setIsNewUser(false);
        }
        setIsLoading(false);
      });

      return unsubscribe;
    };

    let unsubscribe: any;
    initializeAuthListener().then((cleanup) => {
      unsubscribe = cleanup;
      console.log('Auth: onAuthStateChanged listener attached');
    });

    return () => {
      console.log('Auth: Cleaning up auth listener');
      if (unsubscribe) unsubscribe();
    };
  }, []);

  const sendOTP = async (phone: string, appVerifier: any): Promise<string> => {
    try {
      console.log('🚀 Auth: sendOTP called with phone:', phone);
      
      setError(null);
      setIsLoading(true);
      
      // Format phone number using utility function
      const formattedPhone = normalizePhoneNumber(phone);
      
      console.log('📱 Auth: Formatted phone:', formattedPhone);
      setPhoneNumber(formattedPhone);
      
      // Validate phone format
      if (!formattedPhone.startsWith('+')) {
        throw new Error('Số điện thoại phải có mã quốc gia (vd: +84)');
      }
      
      // Try to send real SMS via Firebase
      try {
        const phoneProvider = new PhoneAuthProvider(auth);
        console.log('📤 Auth: Attempting to send SMS via Firebase...');
        console.log('🔑 Auth: Phone:', formattedPhone);
        console.log('🔐 Auth: AppVerifier:', appVerifier ? 'Provided' : 'NULL (using Play Integrity)');
        
        // On Android, appVerifier can be null if Play Integrity is configured
        // Firebase will use Play Integrity API automatically
        const verificationId = await phoneProvider.verifyPhoneNumber(
          formattedPhone,
          appVerifier // null = use Play Integrity on Android
        );
        
        console.log('✅ Auth: SMS sent successfully!');
        console.log('🆔 Auth: Verification ID:', verificationId);
        
        setVerificationId(verificationId);
        setIsLoading(false);
        
        return verificationId;
        
      } catch (firebaseErr: any) {
        console.error('❌ Auth: Firebase SMS error:', firebaseErr);
        console.error('📋 Auth: Error code:', firebaseErr.code);
        console.error('📝 Auth: Error message:', firebaseErr.message);
        console.error('🔍 Auth: Full error:', JSON.stringify(firebaseErr, null, 2));
        
        // Check specific error codes
        if (firebaseErr.code === 'auth/invalid-app-credential' || 
            firebaseErr.code === 'auth/missing-client-identifier' ||
            firebaseErr.code === 'auth/app-not-authorized') {
          console.warn('⚠️  Firebase configuration issue detected');
          console.warn('💡 Possible causes:');
          console.warn('   1. SHA-256 certificate not added to Firebase Console');
          console.warn('   2. google-services.json outdated');
          console.warn('   3. App not authorized in Firebase Console');
          console.warn('   4. Running in development mode without proper config');
          console.warn('');
          console.warn('🔧 Using mock OTP for development...');
          
          // Use mock mode for development
          const mockVerificationId = 'mock-verification-' + Date.now();
          setVerificationId(mockVerificationId);
          setIsLoading(false);
          
          console.log('✅ Mock OTP mode activated');
          console.log('🔑 Use OTP: 123456 to login');
          
          return mockVerificationId;
        }
        
        // For test numbers, use mock mode
        const testNumbers = ['+84945876079', '+16505551234'];
        const cleanedPhone = formattedPhone.replace(/\s/g, '');
        
        if (testNumbers.some(num => cleanedPhone.includes(num.replace(/\s/g, '')))) {
          console.warn('🧪 Test phone number detected, using mock OTP');
          const mockVerificationId = 'mock-verification-' + Date.now();
          setVerificationId(mockVerificationId);
          setIsLoading(false);
          return mockVerificationId;
        }
        
        // For real numbers with errors, throw
        throw firebaseErr;
      }
      
    } catch (err: any) {
      console.error('❌ Auth: sendOTP fatal error:', err);
      
      // Use Vietnamese error messages
      let errorMessage = getVietnameseAuthError(err.code);
      
      // Add helpful hints for common issues
      if (err.code === 'auth/invalid-phone-number') {
        errorMessage += '\n\nVui lòng nhập đầy đủ mã quốc gia (vd: +84 cho Việt Nam)';
      }
      
      if (!errorMessage || errorMessage.includes('Đã có lỗi')) {
        errorMessage = err.message || 'Không thể gửi mã OTP';
      }
      
      setError(errorMessage);
      setIsLoading(false);
      throw err;
    }
  };

  const verifyOTP = async (otp: string): Promise<any> => {
    try {
      console.log('Auth: verifyOTP called with OTP length:', otp.length);
      setError(null);
      setIsLoading(true);

      if (!verificationId) {
        throw new Error('Verification ID not found. Please send OTP first.');
      }

      // Check if this is a mock verification ID (development mode)
      if (verificationId.startsWith('mock-')) {
        console.warn('⚠️  Using mock OTP verification');
        
        // Accept test OTP
        if (otp !== TEST_OTP && otp !== '123456') {
          setError('Invalid OTP. Test OTP: 123456');
          setIsLoading(false);
          throw new Error('Invalid OTP');
        }
        
        // Create mock user
        const mockUid = 'mock-user-' + Date.now();
        const authUser: AuthUser = {
          uid: mockUid,
          phoneNumber: phoneNumber,
          email: null,
          displayName: null,
        };
        
        setUser(authUser);
        mockUserDatabase[mockUid] = {
          uid: mockUid,
          phoneNumber: phoneNumber,
        };
        
        await checkUserExists(mockUid);
        setIsLoading(false);
        
        return { uid: mockUid };
      }

      // Real Firebase verification
      const credential = PhoneAuthProvider.credential(verificationId, otp);
      const userCredential = await signInWithCredential(auth, credential);
      console.log('Auth: User signed in successfully:', userCredential.user.uid);
      
      // Create auth user object
      const authUser: AuthUser = {
        uid: userCredential.user.uid,
        phoneNumber: userCredential.user.phoneNumber,
        email: userCredential.user.email,
        displayName: userCredential.user.displayName,
      };
      
      setUser(authUser);
      
      // Check if user exists in Firestore
      await checkUserExists(userCredential.user.uid);
      
      setIsLoading(false);
      return userCredential.user;
    } catch (err: any) {
      console.error('Auth: verifyOTP error:', err);
      console.error('Auth: Error code:', err.code);
      
      // Use Vietnamese error messages
      const errorMessage = getVietnameseAuthError(err.code) || err.message || 'Mã OTP không hợp lệ';
      
      setError(errorMessage);
      setIsLoading(false);
      throw err;
    }
  };

  const updateUserProfile = async (profile: Partial<UserProfile>) => {
    try {
      if (!user) {
        throw new Error('No user logged in');
      }

      console.log('Auth: updateUserProfile called for UID:', user.uid);
      
      const updatedProfile: UserProfile = {
        uid: user.uid,
        phoneNumber: phoneNumber,
        ...profile,
        createdAt: userProfile?.createdAt || new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };

      try {
        // Try to save to Firestore
        const userDocRef = doc(db, 'users', user.uid);
        await setDoc(userDocRef, updatedProfile);
        console.log('Auth: User profile saved to Firestore');
      } catch (firestoreErr) {
        console.log('Auth: Firestore unavailable, saving to mock database');
        mockUserDatabase[user.uid] = updatedProfile;
      }
      
      setUserProfile(updatedProfile);
      setIsNewUser(false);
    } catch (err: any) {
      console.error('Auth: Error updating user profile:', err);
      setError(err.message || 'Failed to update profile');
      throw err;
    }
  };

  const signOutUser = async () => {
    try {
      console.log('Auth: signOutUser called');
      setError(null);
      await signOut(auth);
      console.log('Auth: Firebase signOut successful');
      setUser(null);
      setUserProfile(null);
      setPhoneNumber('');
      setVerificationId(null);
      setIsNewUser(false);
      console.log('Auth: State cleared - user should be null now');
    } catch (err: any) {
      console.error('Auth: signOut error:', err);
      setError(err.message || 'Failed to sign out');
      throw err;
    }
  };

  const value: AuthContextType = {
    user,
    userProfile,
    isLoading,
    isSignedIn: !!user,
    isNewUser,
    phoneNumber,
    setPhoneNumber,
    verificationId,
    setVerificationId,
    sendOTP,
    verifyOTP,
    signOutUser,
    updateUserProfile,
    error,
    setError,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
