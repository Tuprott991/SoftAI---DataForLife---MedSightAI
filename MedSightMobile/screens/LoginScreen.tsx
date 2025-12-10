import React, { useState, useRef } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  TouchableOpacity,
  TextInput as RNTextInput,
  Alert,
} from 'react-native';
import { TextInput, Button, Text, ActivityIndicator, HelperText } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import { useRouter } from 'expo-router';
import { useAuth } from '@/context/AuthContext';
import { app } from '@/services/firebase';
import { 
  useOTPTimer, 
  formatPhoneDisplay, 
  getVietnameseAuthError,
  validatePhoneNumber,
  formatCountdown 
} from '@/utils/otp-utils';

// Safely import ReCAPTCHA with fallback
let FirebaseRecaptchaVerifierModal: any = null;
let ReCAPTCHAAvailable = true;
try {
  const module = require('expo-firebase-recaptcha');
  FirebaseRecaptchaVerifierModal = module.FirebaseRecaptchaVerifierModal;
} catch (err) {
  console.warn('⚠️  ReCAPTCHA not available, using fallback verification');
  ReCAPTCHAAvailable = false;
  // Fallback: render nothing
  FirebaseRecaptchaVerifierModal = () => null;
}

export default function LoginScreen() {
  const router = useRouter();
  const { sendOTP, verifyOTP, isLoading, error, setError } = useAuth();
  
  const [phoneNumber, setPhoneNumber] = useState('');
  const [step, setStep] = useState<'phone' | 'otp'>('phone');
  const [otp, setOtp] = useState(['', '', '', '', '', '']);
  const [isPhoneValid, setIsPhoneValid] = useState(false);
  const otpInputRefs = useRef<RNTextInput[]>([]);
  const recaptchaVerifier = useRef(null);
  
  // OTP countdown timer for rate limiting
  const { countdown, canResend, startCountdown } = useOTPTimer(60);

  const handlePhoneChange = (text: string) => {
    setPhoneNumber(text);
    setIsPhoneValid(validatePhoneNumber(text));
    if (error) setError(null);
  };

  const handleSendOTP = async () => {
    try {
      console.log('LoginScreen: handleSendOTP called with phone:', phoneNumber);
      
      // Check if ReCAPTCHA is available
      if (ReCAPTCHAAvailable && !recaptchaVerifier.current) {
        Alert.alert('Lỗi', 'ReCAPTCHA chưa sẵn sàng. Vui lòng đợi...');
        return;
      }
      
      setError(null);
      console.log('LoginScreen: Sending OTP...');
      
      // Pass recaptchaVerifier if available, otherwise pass null for fallback
      const verifier = ReCAPTCHAAvailable ? recaptchaVerifier.current : null;
      await sendOTP(phoneNumber, verifier);
      
      console.log('LoginScreen: OTP sent successfully, moving to OTP step');
      setStep('otp');
      startCountdown(); // Start 60s countdown for rate limiting
      Alert.alert('Thành công', `Mã OTP đã được gửi đến ${formatPhoneDisplay(phoneNumber)}`);
    } catch (err: any) {
      console.error('LoginScreen: Error sending OTP:', err);
      const errorMsg = getVietnameseAuthError(err.code) || err.message || 'Không thể gửi mã OTP';
      setError(errorMsg);
      Alert.alert('Lỗi', errorMsg);
    }
  };

  const handleOtpChange = (text: string, index: number) => {
    if (text.length <= 1 && /^\d*$/.test(text)) {
      const newOtp = [...otp];
      newOtp[index] = text;
      setOtp(newOtp);

      // Auto-focus to next field
      if (text && index < 5) {
        otpInputRefs.current[index + 1]?.focus();
      }

      if (error) setError(null);
    }
  };

  const handleOtpKeyPress = (key: string, index: number) => {
    if (key === 'Backspace' && !otp[index] && index > 0) {
      otpInputRefs.current[index - 1]?.focus();
      const newOtp = [...otp];
      newOtp[index - 1] = '';
      setOtp(newOtp);
    }
  };

  const handleVerifyOTP = async () => {
    try {
      console.log('LoginScreen: handleVerifyOTP called');
      setError(null);
      const otpCode = otp.join('');
      if (otpCode.length !== 6) {
        console.warn('LoginScreen: Invalid OTP length:', otpCode.length);
        setError('Please enter 6-digit OTP');
        return;
      }
      console.log('LoginScreen: Verifying OTP code');
      await verifyOTP(otpCode);
      console.log('LoginScreen: OTP verified successfully');
      // Navigation will be handled by the auth state change
    } catch (err) {
      console.error('LoginScreen: Error verifying OTP:', err);
    }
  };

  return (
    <LinearGradient
      colors={['#0A0E27', '#1A1F3A', '#0F3460', '#16213E']}
      locations={[0, 0.3, 0.7, 1]}
      style={styles.gradient}
    >
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.container}
      >
        {/* ReCAPTCHA Verifier Modal - REQUIRED for Firebase Phone Auth */}
        <FirebaseRecaptchaVerifierModal
          ref={recaptchaVerifier}
          firebaseConfig={app.options}
        />
        
        <ScrollView contentContainerStyle={styles.scrollContent}>
          {/* Medical/X-ray themed decorative overlay */}
          <View style={styles.decorativeOverlay}>
            <View style={[styles.scanLine, { top: '10%' }]} />
            <View style={[styles.scanLine, { top: '35%' }]} />
            <View style={[styles.scanLine, { top: '60%' }]} />
            <View style={[styles.scanLine, { top: '85%' }]} />
          </View>

          {/* Header */}
          <View style={styles.header}>
            <View style={styles.iconContainer}>
              <Text style={styles.iconText}>⚕️</Text>
            </View>
            <Text variant="displaySmall" style={styles.title}>
              MedSight AI
            </Text>
            <Text variant="bodyLarge" style={styles.subtitle}>
              {step === 'phone' ? 'Đăng nhập' : 'Xác minh OTP'}
            </Text>
            <Text variant="bodySmall" style={styles.subtitleSecondary}>
              X-Ray Diagnosis System
            </Text>
          </View>

        {/* Content */}
        <View style={styles.content}>
          {step === 'phone' ? (
            <>
              {/* Phone Number Input */}
              <Text variant="labelLarge" style={styles.label}>
                Số điện thoại
              </Text>
              <TextInput
                label="Nhập số điện thoại (+84...)" 
                value={phoneNumber}
                onChangeText={handlePhoneChange}
                keyboardType="phone-pad"
                mode="outlined"
                style={styles.input}
                textColor="#FFFFFF"
                placeholderTextColor="#9BA1A6"
                editable={!isLoading}
                outlineColor="#14B8A6"
                activeOutlineColor="#0D9488"
              />
              {phoneNumber && !isPhoneValid && (
                <HelperText type="error" visible={true}>
                  Số điện thoại không hợp lệ
                </HelperText>
              )}
              <Text variant="bodySmall" style={styles.infoText}>
                Chúng tôi sẽ gửi mã xác minh OTP đến số điện thoại này
              </Text>

              {/* Error Message */}
              {error && (
                <View style={styles.errorBox}>
                  <Text style={styles.errorText}>{error}</Text>
                </View>
              )}

              {/* Send OTP Button */}
              <Button
                mode="contained"
                onPress={handleSendOTP}
                disabled={!isPhoneValid || isLoading}
                loading={isLoading}
                style={styles.button}
                buttonColor="#14B8A6"
              >
                {isLoading ? 'Đang gửi...' : 'Gửi mã OTP'}
              </Button>

              {/* Info Text */}
              <View style={styles.footer}>
                <Text variant="bodySmall" style={styles.infoFooter}>
                  Nhập số điện thoại để bắt đầu. Nếu tài khoản chưa tồn tại, nó sẽ được tạo tự động.
                </Text>
              </View>
            </>
          ) : (
            <>
              {/* OTP Input */}
              <Text variant="labelLarge" style={styles.label}>
                Mã xác minh (6 chữ số)
              </Text>
              <Text variant="bodySmall" style={styles.infoText}>
                Nhập mã xác minh được gửi đến {formatPhoneDisplay(phoneNumber)}
              </Text>

              <View style={styles.otpContainer}>
                {otp.map((digit, index) => (
                  <RNTextInput
                    key={index}
                    ref={(ref) => {
                      if (ref) otpInputRefs.current[index] = ref;
                    }}
                    style={styles.otpInput}
                    maxLength={1}
                    keyboardType="number-pad"
                    value={digit}
                    onChangeText={(text) => handleOtpChange(text, index)}
                    onKeyPress={({ nativeEvent }) =>
                      handleOtpKeyPress(nativeEvent.key, index)
                    }
                    editable={!isLoading}
                    placeholderTextColor="#CCC"
                  />
                ))}
              </View>

              {/* Error Message */}
              {error && (
                <View style={styles.errorBox}>
                  <Text style={styles.errorText}>{error}</Text>
                </View>
              )}

              {/* Verify Button */}
              <Button
                mode="contained"
                onPress={handleVerifyOTP}
                disabled={otp.some((o) => !o) || isLoading}
                loading={isLoading}
                style={styles.button}
                buttonColor="#14B8A6"
              >
                {isLoading ? 'Đang xác minh...' : 'Xác minh'}
              </Button>

              {/* Back Button */}
              <Button
                mode="outlined"
                onPress={() => {
                  setStep('phone');
                  setOtp(['', '', '', '', '', '']);
                  setError(null);
                }}
                style={styles.backButton}
                textColor="#14B8A6"
                disabled={isLoading}
              >
                Quay lại
              </Button>

              {/* Resend OTP */}
              <View style={styles.footer}>
                <Text variant="bodySmall" style={styles.infoFooter}>Không nhận được mã? </Text>
                <TouchableOpacity
                  onPress={handleSendOTP}
                  disabled={isLoading || !canResend}
                >
                  <Text
                    variant="bodySmall"
                    style={[styles.linkText, (!canResend || isLoading) && styles.linkTextDisabled]}
                  >
                    {canResend ? 'Gửi lại' : `Gửi lại sau ${formatCountdown(countdown)}`}
                  </Text>
                </TouchableOpacity>
              </View>
            </>
          )}
        </View>

        {/* Loading Indicator */}
        {isLoading && (
          <View style={styles.loadingOverlay}>
            <ActivityIndicator animating={true} size="large" color="#14B8A6" />
          </View>
        )}
      </ScrollView>
    </KeyboardAvoidingView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  gradient: {
    flex: 1,
  },
  container: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    padding: 20,
  },
  decorativeOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    opacity: 0.1,
  },
  scanLine: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: '#14B8A6',
    shadowColor: '#14B8A6',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 10,
  },
  header: {
    marginBottom: 40,
    alignItems: 'center',
  },
  iconContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(20, 184, 166, 0.15)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
    borderWidth: 2,
    borderColor: '#14B8A6',
  },
  iconText: {
    fontSize: 40,
  },
  title: {
    fontWeight: 'bold',
    color: '#14B8A6',
    marginBottom: 8,
    textShadowColor: 'rgba(20, 184, 166, 0.5)',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 10,
  },
  subtitle: {
    color: '#ECEDEE',
    textAlign: 'center',
    fontWeight: '600',
  },
  subtitleSecondary: {
    color: '#9BA1A6',
    textAlign: 'center',
    marginTop: 4,
    fontStyle: 'italic',
  },
  content: {
    marginBottom: 20,
  },
  label: {
    color: '#ECEDEE',
    marginBottom: 8,
    fontWeight: '600',
  },
  input: {
    marginBottom: 12,
    backgroundColor: 'rgba(31, 41, 55, 0.7)',
  },
  infoText: {
    color: '#9BA1A6',
    marginBottom: 16,
    marginTop: -8,
  },
  button: {
    marginVertical: 16,
    paddingVertical: 8,
  },
  backButton: {
    marginVertical: 8,
    borderColor: '#14B8A6',
  },
  otpContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginVertical: 24,
  },
  otpInput: {
    width: 48,
    height: 56,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#14B8A6',
    backgroundColor: 'rgba(31, 41, 55, 0.7)',
    color: '#ECEDEE',
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 24,
  },
  linkText: {
    color: '#14B8A6',
    fontWeight: '600',
    textDecorationLine: 'underline',
  },
  linkTextDisabled: {
    color: '#757575',
    textDecorationLine: 'none',
  },
  infoFooter: {
    color: '#757575',
    textAlign: 'center',
    marginTop: 16,
  },
  errorBox: {
    backgroundColor: '#DC2626',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    marginBottom: 16,
  },
  errorText: {
    color: '#FFFFFF',
    fontSize: 14,
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
});
