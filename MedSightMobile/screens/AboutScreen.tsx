import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  SafeAreaView,
} from 'react-native';
import Constants from 'expo-constants';

export default function AboutScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.content}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.appName}>MedSight Mobile</Text>
          <Text style={styles.version}>Version {Constants.expoConfig?.version || '1.0.0'}</Text>
          <Text style={styles.tagline}>Chẩn đoán Y khoa thông minh với AI</Text>
        </View>

        {/* Feature List */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>✨ Tính năng</Text>
          <View style={styles.featureList}>
            <FeatureItem 
              icon="🔐" 
              title="Đăng nhập OTP" 
              description="Xác thực an toàn qua số điện thoại"
            />
            <FeatureItem 
              icon="📋" 
              title="Lịch sử khám" 
              description="Theo dõi lịch sử chẩn đoán của bạn"
            />
            <FeatureItem 
              icon="🤖" 
              title="AI Phân tích" 
              description="Phân tích hình ảnh y khoa bằng AI"
            />
            <FeatureItem 
              icon="🔔" 
              title="Thông báo real-time" 
              description="Nhận cập nhật từ bác sĩ ngay lập tức"
            />
            <FeatureItem 
              icon="🔄" 
              title="OTA Updates" 
              description="Tự động cập nhật tính năng mới"
            />
          </View>
        </View>

        {/* Tech Stack */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🛠️ Công nghệ</Text>
          <View style={styles.techStack}>
            <TechBadge name="React Native" />
            <TechBadge name="Expo" />
            <TechBadge name="Firebase" />
            <TechBadge name="TypeScript" />
            <TechBadge name="FastAPI" />
            <TechBadge name="PostgreSQL" />
          </View>
        </View>

        {/* Info */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>ℹ️ Thông tin</Text>
          <InfoRow label="Developer" value="SoftAI Team" />
          <InfoRow label="Build" value={Constants.expoConfig?.extra?.eas?.projectId?.slice(0, 8) || 'dev'} />
          <InfoRow label="Platform" value="Android" />
        </View>

        {/* Footer */}
        <View style={styles.footer}>
          <Text style={styles.footerText}>
            © 2025 MedSight AI. All rights reserved.
          </Text>
          <Text style={styles.footerSubtext}>
            🚀 Powered by OTA Updates - No reinstall needed!
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

// Helper Components
function FeatureItem({ icon, title, description }: { icon: string; title: string; description: string }) {
  return (
    <View style={styles.featureItem}>
      <Text style={styles.featureIcon}>{icon}</Text>
      <View style={styles.featureContent}>
        <Text style={styles.featureTitle}>{title}</Text>
        <Text style={styles.featureDescription}>{description}</Text>
      </View>
    </View>
  );
}

function TechBadge({ name }: { name: string }) {
  return (
    <View style={styles.techBadge}>
      <Text style={styles.techBadgeText}>{name}</Text>
    </View>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.infoRow}>
      <Text style={styles.infoLabel}>{label}:</Text>
      <Text style={styles.infoValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: 24,
    paddingBottom: 40,
  },
  header: {
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 32,
  },
  appName: {
    fontSize: 32,
    fontWeight: '800',
    color: '#14B8A6',
    marginBottom: 8,
  },
  version: {
    fontSize: 16,
    color: '#6B7280',
    marginBottom: 12,
  },
  tagline: {
    fontSize: 16,
    color: '#4B5563',
    fontStyle: 'italic',
    textAlign: 'center',
  },
  section: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1F2937',
    marginBottom: 16,
  },
  featureList: {
    gap: 16,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  featureIcon: {
    fontSize: 28,
    marginRight: 12,
  },
  featureContent: {
    flex: 1,
  },
  featureTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 4,
  },
  featureDescription: {
    fontSize: 14,
    color: '#6B7280',
    lineHeight: 20,
  },
  techStack: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  techBadge: {
    backgroundColor: '#E0F2FE',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#0EA5E9',
  },
  techBadgeText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#0369A1',
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  infoLabel: {
    fontSize: 16,
    color: '#6B7280',
    fontWeight: '500',
  },
  infoValue: {
    fontSize: 16,
    color: '#1F2937',
    fontWeight: '600',
  },
  footer: {
    marginTop: 32,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 14,
    color: '#9CA3AF',
    marginBottom: 8,
  },
  footerSubtext: {
    fontSize: 12,
    color: '#14B8A6',
    fontWeight: '600',
  },
});
