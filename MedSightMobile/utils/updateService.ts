/**
 * OTA Update Service
 * Automatically checks and applies Expo Updates
 */
import * as Updates from 'expo-updates';
import { Alert } from 'react-native';

export class UpdateService {
  /**
   * Check for available updates and apply them
   */
  static async checkForUpdates(showAlert = false): Promise<boolean> {
    try {
      // Skip in development mode
      if (__DEV__) {
        console.log('⚠️  Update check skipped in development mode');
        return false;
      }

      console.log('🔍 Checking for updates...');
      
      const update = await Updates.checkForUpdateAsync();
      
      if (update.isAvailable) {
        console.log('✅ Update available! Downloading...');
        
        if (showAlert) {
          Alert.alert(
            '🚀 Cập nhật mới',
            'Đang tải phiên bản mới của ứng dụng...',
            [{ text: 'OK' }]
          );
        }
        
        // Download the update
        await Updates.fetchUpdateAsync();
        
        console.log('✅ Update downloaded! Reloading app...');
        
        // Reload the app to apply update
        await Updates.reloadAsync();
        
        return true;
      } else {
        console.log('✓ App is up to date');
        return false;
      }
    } catch (error: any) {
      console.error('❌ Error checking for updates:', error);
      
      if (showAlert) {
        Alert.alert(
          'Lỗi cập nhật',
          'Không thể kiểm tra cập nhật. Vui lòng thử lại sau.',
          [{ text: 'OK' }]
        );
      }
      
      return false;
    }
  }

  /**
   * Check for updates with user prompt
   */
  static async checkForUpdatesWithPrompt(): Promise<void> {
    try {
      if (__DEV__) {
        Alert.alert('Development Mode', 'Updates are disabled in development mode');
        return;
      }

      const update = await Updates.checkForUpdateAsync();
      
      if (update.isAvailable) {
        Alert.alert(
          '🎉 Cập nhật có sẵn',
          'Có phiên bản mới của ứng dụng. Bạn có muốn cập nhật ngay không?',
          [
            {
              text: 'Để sau',
              style: 'cancel',
            },
            {
              text: 'Cập nhật',
              onPress: async () => {
                try {
                  await Updates.fetchUpdateAsync();
                  Alert.alert(
                    'Thành công',
                    'Ứng dụng sẽ khởi động lại để áp dụng cập nhật',
                    [
                      {
                        text: 'OK',
                        onPress: () => Updates.reloadAsync(),
                      },
                    ]
                  );
                } catch (error) {
                  Alert.alert('Lỗi', 'Không thể tải cập nhật');
                }
              },
            },
          ]
        );
      } else {
        Alert.alert('✓ Đã cập nhật', 'Bạn đang sử dụng phiên bản mới nhất');
      }
    } catch (error) {
      Alert.alert('Lỗi', 'Không thể kiểm tra cập nhật');
    }
  }

  /**
   * Get current update info
   */
  static async getCurrentUpdateInfo() {
    try {
      if (!Updates.isEnabled) {
        return {
          isEnabled: false,
          updateId: null,
          channel: null,
          runtimeVersion: null,
        };
      }

      const updateId = Updates.updateId;
      const channel = Updates.channel;
      const runtimeVersion = Updates.runtimeVersion;

      return {
        isEnabled: true,
        updateId,
        channel,
        runtimeVersion,
      };
    } catch (error) {
      console.error('Error getting update info:', error);
      return null;
    }
  }
}
