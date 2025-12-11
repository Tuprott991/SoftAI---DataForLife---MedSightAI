import messaging from '@react-native-firebase/messaging';
import { Platform, Alert } from 'react-native';
import { db } from './firebase';
import { doc, setDoc, updateDoc } from 'firebase/firestore';

/**
 * Request notification permissions
 */
export const requestNotificationPermission = async (): Promise<boolean> => {
  try {
    const authStatus = await messaging().requestPermission();
    const enabled =
      authStatus === messaging.AuthorizationStatus.AUTHORIZED ||
      authStatus === messaging.AuthorizationStatus.PROVISIONAL;

    if (enabled) {
      console.log('✅ Notification permission granted:', authStatus);
      return true;
    } else {
      console.log('❌ Notification permission denied');
      return false;
    }
  } catch (error) {
    console.error('❌ Error requesting notification permission:', error);
    return false;
  }
};

/**
 * Get FCM token
 */
export const getFCMToken = async (): Promise<string | null> => {
  try {
    // Check if permission is granted
    const hasPermission = await requestNotificationPermission();
    if (!hasPermission) {
      console.warn('⚠️ Notification permission not granted');
      return null;
    }

    // Get FCM token
    const token = await messaging().getToken();
    console.log('✅ FCM Token:', token);
    return token;
  } catch (error) {
    console.error('❌ Error getting FCM token:', error);
    return null;
  }
};

/**
 * Save FCM token to Firestore
 */
export const saveFCMTokenToFirestore = async (
  userId: string,
  token: string
): Promise<void> => {
  try {
    const userRef = doc(db, 'users', userId);
    
    await updateDoc(userRef, {
      fcmToken: token,
      fcmTokenUpdatedAt: new Date().toISOString(),
      platform: Platform.OS,
      updatedAt: new Date().toISOString(),
    });

    console.log('✅ FCM token saved to Firestore');
  } catch (error) {
    console.error('❌ Error saving FCM token to Firestore:', error);
    throw error;
  }
};

/**
 * Initialize FCM token and save to DB
 */
export const initializeFCMToken = async (userId: string): Promise<(() => void) | undefined> => {
  try {
    // Get FCM token
    const token = await getFCMToken();
    
    if (!token) {
      console.warn('⚠️ No FCM token received');
      return;
    }

    // Save to Firestore
    await saveFCMTokenToFirestore(userId, token);

    // Listen for token refresh
    const unsubscribe = messaging().onTokenRefresh(async (newToken) => {
      console.log('🔄 FCM Token refreshed:', newToken);
      await saveFCMTokenToFirestore(userId, newToken);
    });

    // Return unsubscribe function (optional: store it if needed)
    return unsubscribe;
  } catch (error) {
    console.error('❌ Error initializing FCM token:', error);
  }
};

/**
 * Handle foreground notifications
 */
export const setupForegroundNotificationHandler = () => {
  const unsubscribe = messaging().onMessage(async (remoteMessage) => {
    console.log('📬 Foreground notification received:', remoteMessage);

    // Show alert for foreground notification
    if (remoteMessage.notification) {
      Alert.alert(
        remoteMessage.notification.title || 'Thông báo',
        remoteMessage.notification.body || '',
        [{ text: 'OK' }]
      );
    }

    // Handle custom data
    if (remoteMessage.data) {
      console.log('📦 Notification data:', remoteMessage.data);
      // Add custom handling logic here
    }
  });

  return unsubscribe;
};

/**
 * Handle background/quit state notifications
 */
export const setupBackgroundNotificationHandler = () => {
  // Handle notification when app is opened from background/quit state
  messaging().onNotificationOpenedApp((remoteMessage) => {
    console.log('📬 Background notification opened:', remoteMessage);
    
    // Navigate to specific screen based on notification data
    if (remoteMessage.data) {
      console.log('📦 Notification data:', remoteMessage.data);
      // Add navigation logic here
      // Example: router.push('/notifications')
    }
  });

  // Handle notification when app is opened from quit state
  messaging()
    .getInitialNotification()
    .then((remoteMessage) => {
      if (remoteMessage) {
        console.log('📬 Quit state notification opened:', remoteMessage);
        
        // Navigate to specific screen based on notification data
        if (remoteMessage.data) {
          console.log('📦 Notification data:', remoteMessage.data);
          // Add navigation logic here
        }
      }
    });
};

/**
 * Request permission and setup all notification handlers
 */
export const setupNotifications = async (userId: string) => {
  try {
    console.log('🔔 Setting up notifications for user:', userId);

    // Initialize FCM token
    await initializeFCMToken(userId);

    // Setup foreground handler
    const unsubscribeForeground = setupForegroundNotificationHandler();

    // Setup background handler
    setupBackgroundNotificationHandler();

    console.log('✅ Notifications setup complete');

    return () => {
      unsubscribeForeground();
    };
  } catch (error) {
    console.error('❌ Error setting up notifications:', error);
  }
};
