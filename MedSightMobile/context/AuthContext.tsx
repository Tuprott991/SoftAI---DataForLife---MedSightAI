import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// Mock storage for users (simulates Firestore)
const mockUserDatabase: Record<string, any> = {};

// Test phone numbers and OTP configuration
const TEST_PHONES = ['+84 945 876 079', '+1 650-555-1234'];
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
  verificationId: any;
  setVerificationId: (id: any) => void;
  sendOTP: (phoneNumber: string) => Promise<void>;
  verifyOTP: (otp: string) => Promise<void>;
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
  const [verificationId, setVerificationId] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Check if user exists in mock database (simulates Firestore)
  const checkUserExists = async (uid: string): Promise<boolean> => {
    try {
      console.log('Auth: Checking if user exists, UID:', uid);
      
      if (mockUserDatabase[uid]) {
        console.log('Auth: User found in mock database');
        setUserProfile(mockUserDatabase[uid]);
        setIsNewUser(false);
        return true;
      } else {
        console.log('Auth: User NOT found in mock database - new user');
        setIsNewUser(true);
        return false;
      }
    } catch (err) {
      console.error('Auth: Error checking user:', err);
      return false;
    }
  };

  useEffect(() => {
    const initializeAuthListener = async () => {
      try {
        // Clear any cached user on app startup (mock)
        console.log('Auth: Cleared cached sessions on startup');
      } catch (err) {
        console.log('Auth: No active session to clear');
      }
      
      // Simulate auth state check
      setIsLoading(false);
      console.log('Auth: onAuthStateChanged listener initialized (mock)');
    };

    initializeAuthListener();
  }, []);

  const sendOTP = async (phone: string) => {
    try {
      console.log('Auth: sendOTP called with phone:', phone);
      setError(null);
      setIsLoading(true);
      
      // Format phone number
      const formattedPhone = phone.startsWith('+') ? phone : `+${phone}`;
      console.log('Auth: Formatted phone:', formattedPhone);
      
      // Check if this is a test phone number
      const normalizedPhone = formattedPhone.replace(/\s/g, '');
      const normalizedTestPhones = TEST_PHONES.map(p => p.replace(/\s/g, ''));
      const isTestPhone = normalizedTestPhones.includes(normalizedPhone);
      
      if (isTestPhone) {
        console.log('Auth: Test phone detected, using mock OTP verification');
        // For test phones, use mock verification
        const mockVerificationId = 'mock-verification-' + Date.now();
        console.log('Auth: Mock verification ID generated:', mockVerificationId);
        setVerificationId(mockVerificationId);
        setPhoneNumber(formattedPhone);
        setIsLoading(false);
        return;
      }
      
      // For production, real phone numbers would need backend OTP service
      console.log('Auth: Real phone number - using test OTP 123456 for demo');
      const mockVerificationId = 'mock-verification-' + Date.now();
      setVerificationId(mockVerificationId);
      setPhoneNumber(formattedPhone);
      setIsLoading(false);
      
    } catch (err: any) {
      console.error('Auth: sendOTP error:', err);
      setError(err.message || 'Failed to send OTP');
      setIsLoading(false);
      throw err;
    }
  };

  const verifyOTP = async (otp: string) => {
    try {
      console.log('Auth: verifyOTP called with OTP:', otp);
      setError(null);
      setIsLoading(true);

      if (!verificationId) {
        console.error('Auth: Verification ID not found');
        throw new Error('Verification ID not found');
      }

      // Check if this is a mock verification (test phone)
      if (verificationId.startsWith('mock-verification-')) {
        console.log('Auth: Mock verification detected');
        
        // For test phones, correct OTP is 123456
        if (otp === TEST_OTP) {
          console.log('Auth: Test OTP verified successfully');
          
          // Create a mock user with the test phone
          const testUid = 'test-user-' + Date.now();
          setUser({
            uid: testUid,
            phoneNumber: phoneNumber,
            email: null,
            displayName: null,
          });
          
          // Check if user exists in Firestore (they won't on first login)
          const exists = await checkUserExists(testUid);
          
          console.log('Auth: Mock user created, OTP verification complete, isNewUser =', !exists);
          setIsLoading(false);
          return;
        } else {
          console.warn('Auth: Incorrect test OTP provided. Expected:', TEST_OTP, 'Got:', otp);
          setError(`Incorrect OTP. For test, use: ${TEST_OTP}`);
          setIsLoading(false);
          throw new Error('Invalid OTP');
        }
      }

      setIsLoading(false);
    } catch (err: any) {
      console.error('Auth: verifyOTP error:', err);
      setError(err.message || 'Invalid OTP');
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

      // Save to mock database (simulates Firestore)
      mockUserDatabase[user.uid] = updatedProfile;
      console.log('Auth: User profile saved to mock database');
      
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
      // Mock sign out - just clear state
      console.log('Auth: Mock sign out successful');
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
