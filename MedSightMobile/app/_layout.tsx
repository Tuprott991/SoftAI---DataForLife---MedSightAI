import { DarkTheme, DefaultTheme, ThemeProvider } from '@react-navigation/native';
import { Stack, useRouter } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import 'react-native-reanimated';
import React, { useEffect } from 'react';
import { ActivityIndicator, View } from 'react-native';
import { AuthProvider, useAuth } from '@/context/AuthContext';
import { useColorScheme } from '@/hooks/use-color-scheme';

export const unstable_settings = {
  anchor: '(tabs)',
};

function RootLayoutNav() {
  const colorScheme = useColorScheme();
  const { isSignedIn, isLoading, user, isNewUser } = useAuth();
  const router = useRouter();

  console.log('RootLayoutNav: isLoading =', isLoading, ', isSignedIn =', isSignedIn, ', isNewUser =', isNewUser, ', user =', user?.uid);

  // Use effect to handle navigation when auth state changes
  useEffect(() => {
    if (!isLoading) {
      console.log('RootLayoutNav: Auth loading complete, isSignedIn =', isSignedIn, ', isNewUser =', isNewUser);
      if (isSignedIn) {
        if (isNewUser) {
          console.log('RootLayoutNav: New user, navigating to profile setup');
          router.replace('/ProfileSetup' as any);
        } else {
          console.log('RootLayoutNav: Existing user, navigating to (tabs)');
          router.replace('/(tabs)' as any);
        }
      } else {
        console.log('RootLayoutNav: User not signed in, navigating to login');
        router.replace('/login' as any);
      }
    }
  }, [isLoading, isSignedIn, isNewUser, router]);

  if (isLoading) {
    console.log('RootLayoutNav: Still loading, showing spinner');
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#151718' }}>
        <ActivityIndicator size="large" color="#14B8A6" />
      </View>
    );
  }

  return (
    <ThemeProvider value={colorScheme === 'dark' ? DarkTheme : DefaultTheme}>
      <Stack 
        screenOptions={{ headerShown: false }}
      >
        {/* Auth Screens */}
        <Stack.Screen name="login" />
        
        {/* Profile Setup Screen */}
        <Stack.Screen name="ProfileSetup" />
        
        {/* App Screens */}
        <Stack.Screen name="(tabs)" />
        <Stack.Screen
          name="ResultDetail"
          options={{ presentation: 'card' }}
        />
      </Stack>
      <StatusBar style="auto" />
    </ThemeProvider>
  );
}

export default function RootLayout() {
  return (
    <AuthProvider>
      <RootLayoutNav />
    </AuthProvider>
  );
}
