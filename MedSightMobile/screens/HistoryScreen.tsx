import React, { useState, useCallback } from 'react';
import {
  View,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  RefreshControl,
  ScrollView,
  Image,
  ImageSourcePropType,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Searchbar, Card, Text, Badge, Avatar, Divider, Chip } from 'react-native-paper';
import { useAuth } from '@/context/AuthContext';
import { useNavigation } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';

type HeatmapPresetKey = 'focalLeft' | 'diffuseCentral' | 'focalRight' | 'baseline';

type HeatmapPresetConfig = {
  baseColors: [string, string, ...string[]];
  highlight: {
    colors: [string, string, ...string[]];
    size: { width: number; height: number };
    position: { top: number; left: number };
    start: { x: number; y: number };
    end: { x: number; y: number };
    rotate?: string;
  };
  bloom?: {
    colors: [string, string, ...string[]];
    size: { width: number; height: number };
    position: { top: number; left: number };
    opacity: number;
  };
  streaks?: { top: number; opacity: number }[];
};

type CasePreviewImage =
  | {
      id: string;
      type: 'heatmap';
      preset: HeatmapPresetKey;
      label?: string;
    }
  | {
      id: string;
      type: 'asset';
      source: ImageSourcePropType;
      label?: string;
    };

const HEATMAP_PRESETS: Record<HeatmapPresetKey, HeatmapPresetConfig> = {
  focalLeft: {
    baseColors: ['#160F2B', '#1F1740', '#2B2056'],
    highlight: {
      colors: ['rgba(24, 219, 172, 0.9)', 'rgba(64, 184, 244, 0.85)', 'rgba(248, 94, 94, 0.9)'],
      size: { width: 170, height: 200 },
      position: { top: -30, left: -10 },
      start: { x: 0.1, y: 0.2 },
      end: { x: 0.8, y: 0.9 },
      rotate: '-18deg',
    },
    bloom: {
      colors: ['rgba(20, 184, 166, 0.2)', 'rgba(20, 184, 166, 0)'],
      size: { width: 210, height: 240 },
      position: { top: -55, left: -35 },
      opacity: 0.8,
    },
    streaks: [
      { top: 38, opacity: 0.18 },
      { top: 78, opacity: 0.1 },
      { top: 118, opacity: 0.12 },
    ],
  },
  diffuseCentral: {
    baseColors: ['#160F2B', '#1E183F', '#292053'],
    highlight: {
      colors: ['rgba(88, 247, 195, 0.9)', 'rgba(246, 201, 86, 0.85)', 'rgba(242, 97, 118, 0.8)'],
      size: { width: 220, height: 180 },
      position: { top: -20, left: -40 },
      start: { x: 0.3, y: 0 },
      end: { x: 0.5, y: 1 },
    },
    bloom: {
      colors: ['rgba(246, 201, 86, 0.15)', 'rgba(246, 201, 86, 0)'],
      size: { width: 240, height: 260 },
      position: { top: -50, left: -50 },
      opacity: 0.9,
    },
    streaks: [
      { top: 32, opacity: 0.14 },
      { top: 72, opacity: 0.12 },
      { top: 112, opacity: 0.16 },
    ],
  },
  focalRight: {
    baseColors: ['#160F2B', '#1C1539', '#241C4A'],
    highlight: {
      colors: ['rgba(46, 199, 226, 0.85)', 'rgba(76, 248, 166, 0.9)', 'rgba(248, 112, 112, 0.85)'],
      size: { width: 170, height: 190 },
      position: { top: -32, left: -30 },
      start: { x: 0.2, y: 0.2 },
      end: { x: 0.9, y: 0.8 },
      rotate: '14deg',
    },
    bloom: {
      colors: ['rgba(76, 248, 166, 0.2)', 'rgba(76, 248, 166, 0)'],
      size: { width: 210, height: 220 },
      position: { top: -58, left: -52 },
      opacity: 0.85,
    },
    streaks: [
      { top: 40, opacity: 0.16 },
      { top: 82, opacity: 0.11 },
      { top: 124, opacity: 0.13 },
    ],
  },
  baseline: {
    baseColors: ['#120C24', '#1A1234', '#231945'],
    highlight: {
      colors: ['rgba(70, 235, 210, 0.75)', 'rgba(144, 202, 249, 0.6)', 'rgba(162, 114, 255, 0.6)'],
      size: { width: 160, height: 200 },
      position: { top: -30, left: -10 },
      start: { x: 0.2, y: 0.1 },
      end: { x: 0.6, y: 0.9 },
    },
    streaks: [
      { top: 46, opacity: 0.08 },
      { top: 96, opacity: 0.1 },
      { top: 136, opacity: 0.07 },
    ],
  },
};

type DoctorDiagnosis = {
  doctor_name: string;
  conclusion?: string;
  notes?: string;
  ai_prediction?: string;
};

type HistoryCase = {
  id: string;
  caseId: string;
  created_at: string;
  patient_name: string;
  status: 'pending' | 'completed';
  symptoms: string;
  doctor_diagnosis?: DoctorDiagnosis;
  images: CasePreviewImage[];
};

const HeatmapMock = ({ preset, label }: { preset: HeatmapPresetKey; label?: string }) => {
  const config = HEATMAP_PRESETS[preset];

  return (
    <View style={styles.heatmapItem}>
      <LinearGradient colors={config.baseColors} style={styles.heatmapCanvas}>
        <LinearGradient
          colors={config.highlight.colors}
          start={config.highlight.start}
          end={config.highlight.end}
          style={[
            styles.heatmapHighlight,
            {
              width: config.highlight.size.width,
              height: config.highlight.size.height,
              top: config.highlight.position.top,
              left: config.highlight.position.left,
              borderRadius: Math.max(config.highlight.size.width, config.highlight.size.height),
              transform: config.highlight.rotate ? [{ rotate: config.highlight.rotate }] : undefined,
            },
          ]}
        />

        {config.bloom ? (
          <LinearGradient
            colors={config.bloom.colors}
            start={{ x: 0.5, y: 0 }}
            end={{ x: 0.5, y: 1 }}
            style={[
              styles.heatmapBloom,
              {
                width: config.bloom.size.width,
                height: config.bloom.size.height,
                top: config.bloom.position.top,
                left: config.bloom.position.left,
                opacity: config.bloom.opacity,
                borderRadius: Math.max(config.bloom.size.width, config.bloom.size.height),
              },
            ]}
          />
        ) : null}

        <LinearGradient
          colors={['rgba(255, 255, 255, 0.18)', 'rgba(255, 255, 255, 0)']}
          start={{ x: 0.5, y: 0 }}
          end={{ x: 0.5, y: 1 }}
          style={styles.heatmapChestGlow}
        />

        {config.streaks?.map((streak, index) => (
          <View
            key={`streak-${preset}-${index}`}
            style={[
              styles.heatmapStreak,
              {
                top: streak.top,
                opacity: streak.opacity,
              },
            ]}
          />
        ))}

        <View style={styles.heatmapFrame} />
      </LinearGradient>
      {label ? <Text style={styles.heatmapCaption}>{label}</Text> : null}
    </View>
  );
};

const CasePreview = ({ preview }: { preview: CasePreviewImage }) => {
  if (preview.type === 'asset') {
    return (
      <View style={styles.heatmapItem}>
        <View style={styles.assetFrame}>
          <Image source={preview.source} style={styles.assetImage} resizeMode="cover" />
          <LinearGradient
            colors={['rgba(22, 24, 59, 0)', 'rgba(22, 24, 59, 0.8)'] as [string, string]}
            style={styles.assetOverlay}
          />
          <View style={styles.assetBorder} />
        </View>
        {preview.label ? <Text style={styles.heatmapCaption}>{preview.label}</Text> : null}
      </View>
    );
  }

  return <HeatmapMock preset={preview.preset} label={preview.label} />;
};

// Mock data generator - will be replaced with API calls
const generateMockHistory = (): HistoryCase[] => {
  const timestamp = Date.now();
  return [
    {
      id: `CASE_001_${timestamp}`,
      caseId: 'CASE_001',
      created_at: new Date().toISOString().split('T')[0],
      patient_name: 'Dương Gia Long',
      status: 'pending',
      symptoms: 'Đau ngực trái, khó thở khi vận động mạnh.',
      doctor_diagnosis: {
        doctor_name: 'ThS. BS Lê Thị Hương'
      },
      images: [],
    },
    {
      id: `CASE_002_${timestamp}`,
      caseId: 'CASE_002',
      created_at: new Date(Date.now() - 86400000).toISOString().split('T')[0],
      patient_name: 'Dương Gia Long',
      status: 'completed',
      symptoms: 'Ho Khan kéo dài, nghẹt mũi, đau họng.',
      doctor_diagnosis: {
        doctor_name: 'BS. Nguyễn Văn Tuấn',
        conclusion: 'Viêm phổi',
        notes: 'Sử dụng thuốc xịt mũi và kháng sinh theo đơn.',
        ai_prediction: 'Pneumonia (95.7%), với các dấu hiệu rõ ràng trên ảnh X-quang gồm: tăng đậm độ phổi, mờ vùng phổi và dấu hiệu viêm.',
      },
      images: [
        {
          id: `CASE_002_ORIGINAL_${timestamp}`,
          type: 'asset',
          source: require('@/assets/images/origin.png'),
          label: 'Ảnh X-quang gốc',
        },
        {
          id: `CASE_002_CARDIOMEGALY_${timestamp}`,
          type: 'asset',
          source: require('@/assets/images/Cardiomegaly.png'),
          label: 'Gợi ý Cardiomegaly',
        },
        {
          id: `CASE_002_AORTIC_${timestamp}`,
          type: 'asset',
          source: require('@/assets/images/Aortic_enlargement.png'),
          label: 'Gợi ý Aortic Enlargement',
        },
        {
          id: `CASE_002_INFILTRATION_${timestamp}`,
          type: 'asset',
          source: require('@/assets/images/Infiltration.jpeg'),
          label: 'Gợi ý Infiltration',
        },
        {
          id: `CASE_002_LUNG_OPACITY_${timestamp}`,
          type: 'asset',
          source: require('@/assets/images/lung_opacity.jpeg'),
          label: 'Gợi ý Lung Opacity',
        },
      ],
    },
    {
      id: `CASE_003_${timestamp}`,
      caseId: 'CASE_003',
      created_at: new Date(Date.now() - 172800000).toISOString().split('T')[0],
      patient_name: 'Nguyễn Văn A',
      status: 'completed',
      symptoms: 'Sốt cao trên 39°C, đau đầu.',
      doctor_diagnosis: {
        doctor_name: 'TS. BS Trần Văn C',
        conclusion: 'Cảm cúm mùa A.',
        notes: 'Uống các thuốc hạ sốt, dùng kháng virus theo đơn.',
        ai_prediction: 'Influenza (92.3%)',
      },
      images: [
        {
          id: `CASE_003_OVERVIEW_${timestamp}`,
          type: 'heatmap',
          preset: 'diffuseCentral',
          label: 'Tăng đậm độ vùng trung tâm',
        },
        {
          id: `CASE_003_RIGHT_${timestamp}`,
          type: 'heatmap',
          preset: 'focalRight',
          label: 'Quan sát vùng phổi phải',
        },
      ],
    },
  ];
};

export default function HistoryScreen() {
  const navigation = useNavigation<any>();
  const { user, signOutUser } = useAuth();
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredData, setFilteredData] = useState(generateMockHistory());
  const [refreshing, setRefreshing] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);

  // Search handler
  const onChangeSearch = (query: string) => {
    setSearchQuery(query);
    const results = generateMockHistory().filter(
      (item) =>
        item.symptoms.toLowerCase().includes(query.toLowerCase()) ||
        item.id.toLowerCase().includes(query.toLowerCase()) ||
        item.patient_name.toLowerCase().includes(query.toLowerCase())
    );
    setFilteredData(results);
  };

  // Refresh handler - simulates API call
  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      // Simulate API call delay
      await new Promise((resolve) => setTimeout(resolve, 1500));
      
      // Generate new mock data (in real app, this would be an API call)
      const newData = generateMockHistory();
      setFilteredData(newData);
      
      // Show success message
      console.log('History refreshed successfully');
    } catch (error) {
      console.error('Error refreshing history:', error);
    } finally {
      setRefreshing(false);
    }
  }, []);

  // Load more handler
  const onLoadMore = useCallback(async () => {
    if (isLoadingMore) return;
    
    setIsLoadingMore(true);
    try {
      // Simulate API call delay
      await new Promise((resolve) => setTimeout(resolve, 1000));
      
      // Generate additional mock data
      const additionalData = generateMockHistory();
      setFilteredData((prevData) => [...prevData, ...additionalData]);
    } catch (error) {
      console.error('Error loading more history:', error);
    } finally {
      setIsLoadingMore(false);
    }
  }, [isLoadingMore]);

  // Handle item press - navigate to detail screen
  const handleItemPress = (item: HistoryCase) => {
    navigation.navigate('ResultDetail', { caseData: item });
  };

  // Get status color
  const getStatusColor = (status: string) =>
    status === 'completed' ? '#10B981' : '#F59E0B';

  // Get status text
  const getStatusText = (status: string) =>
    status === 'completed' ? 'Đã có kết quả' : 'Đang chờ bác sĩ';

  // Render history item
  const renderItem = ({ item }: { item: HistoryCase }) => (
    <TouchableOpacity
      onPress={() => handleItemPress(item)}
      activeOpacity={0.8}
    >
      <Card style={styles.card}>
        <Card.Title
          title={`Hồ sơ: ${item.id}`}
          subtitle={`Ngày: ${item.created_at}`}
          left={(props) => (
            <Avatar.Icon
              {...props}
              icon="file-document-outline"
              style={{ backgroundColor: '#14B8A6' }}
            />
          )}
          right={(props) => (
            <View style={{ marginRight: 16 }}>
              <Badge style={{ backgroundColor: getStatusColor(item.status) }}>
                {getStatusText(item.status)}
              </Badge>
            </View>
          )}
        />
        <Divider />
        <Card.Content style={{ paddingTop: 10 }}>
          <Text variant="bodyMedium" numberOfLines={2} style={{ color: '#555' }}>
            <Text style={{ fontWeight: 'bold' }}>Triệu chứng: </Text>
            {item.symptoms}
          </Text>
          {item.status === 'completed' && item.doctor_diagnosis && (
            <Text style={{ marginTop: 5, color: '#14B8A6', fontStyle: 'italic' }}>
              ✓ Bác sĩ {item.doctor_diagnosis.doctor_name} đã phản hồi
            </Text>
          )}
          {item.images.length > 0 && (
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.heatmapList}
              style={styles.heatmapListWrapper}
            >
              {item.images.map((preview) => (
                <CasePreview key={preview.id} preview={preview} />
              ))}
            </ScrollView>
          )}
        </Card.Content>
      </Card>
    </TouchableOpacity>
  );

  // Render empty state
  const renderEmptyState = () => (
    <View style={styles.emptyContainer}>
      <Avatar.Icon
        icon="clipboard-outline"
        size={64}
        style={[styles.emptyIcon, { backgroundColor: '#14B8A6' }]}
      />
      <Text variant="headlineSmall" style={styles.emptyTitle}>
        Không tìm thấy hồ sơ
      </Text>
      <Text variant="bodyMedium" style={styles.emptySubtitle}>
        {searchQuery ? 'Thử điều chỉnh từ khóa tìm kiếm' : 'Bạn chưa có hồ sơ khám'}
      </Text>
    </View>
  );

  // Render footer (loading more indicator)
  const renderFooter = () => {
    if (!isLoadingMore) return null;
    return (
      <View style={styles.loadingFooter}>
        <Text style={styles.loadingText}>Đang tải thêm...</Text>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header Section */}
      <View style={styles.headerSection}>
        <View style={styles.headerTop}>
          <View>
            <Text variant="headlineMedium" style={styles.headerTitle}>
              Lịch sử khám
            </Text>
            <Text style={styles.headerSubtitle}>
              {user?.phoneNumber || 'Bệnh nhân'}
            </Text>
          </View>
          <TouchableOpacity
            style={styles.signOutBtn}
            onPress={async () => {
              try {
                console.log('HistoryScreen: Logout button pressed');
                await signOutUser();
                console.log('HistoryScreen: signOutUser completed');
              } catch (err) {
                console.error('HistoryScreen: Logout error:', err);
              }
            }}
          >
            <Text style={styles.signOutText}>Đăng xuất</Text>
          </TouchableOpacity>
        </View>

        {/* Search Bar */}
        <View style={styles.searchSection}>
          <Searchbar
            placeholder="Tìm theo triệu chứng, mã HS..."
            onChangeText={onChangeSearch}
            value={searchQuery}
            style={styles.searchBar}
            inputStyle={styles.searchInput}
            elevation={1}
            iconColor="#14B8A6"
          />
        </View>

        {/* Filter Chips */}
        {filteredData.length > 0 && (
          <View style={styles.filterSection}>
            <Chip
              selected={false}
              mode="outlined"
              style={styles.chip}
              textStyle={{ fontSize: 12 }}
            >
              Tổng: {filteredData.length}
            </Chip>
            <Chip
              selected={false}
              mode="outlined"
              style={styles.chip}
              textStyle={{ fontSize: 12 }}
            >
              Hoàn thành:{' '}
              {filteredData.filter((item) => item.status === 'completed').length}
            </Chip>
            <Chip
              selected={false}
              mode="outlined"
              style={styles.chip}
              textStyle={{ fontSize: 12 }}
            >
              Chờ:{' '}
              {filteredData.filter((item) => item.status === 'pending').length}
            </Chip>
          </View>
        )}
      </View>

      {/* History List */}
      {filteredData.length > 0 ? (
        <FlatList
          data={filteredData}
          renderItem={renderItem}
          keyExtractor={(item, index) => `${item.id}_${index}`}
          contentContainerStyle={styles.listContent}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              colors={['#14B8A6']}
              tintColor="#14B8A6"
              title="Đang tải lại..."
              titleColor="#ECEDEE"
            />
          }
          onEndReached={onLoadMore}
          onEndReachedThreshold={0.5}
          ListFooterComponent={renderFooter}
          scrollIndicatorInsets={{ right: 1 }}
        />
      ) : (
        <ScrollView
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              colors={['#14B8A6']}
              tintColor="#14B8A6"
              title="Đang tải lại..."
              titleColor="#ECEDEE"
            />
          }
          contentContainerStyle={styles.emptyScrollContent}
        >
          {renderEmptyState()}
        </ScrollView>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F7FA',
  },
  headerSection: {
    backgroundColor: '#FFFFFF',
    paddingTop: 12,
    paddingHorizontal: 16,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  headerTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  headerTitle: {
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#9CA3AF',
  },
  signOutBtn: {
    backgroundColor: '#FEE2E2',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  signOutText: {
    color: '#DC2626',
    fontSize: 12,
    fontWeight: '600',
  },
  searchSection: {
    marginBottom: 12,
  },
  searchBar: {
    backgroundColor: '#F3F4F6',
    borderRadius: 8,
    elevation: 0,
    borderWidth: 1,
    borderColor: '#E5E7EB',
  },
  searchInput: {
    fontSize: 14,
    color: '#1F2937',
  },
  filterSection: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 8,
  },
  chip: {
    borderColor: '#D1D5DB',
    backgroundColor: '#F9FAFB',
  },
  listContent: {
    paddingHorizontal: 10,
    paddingVertical: 12,
  },
  card: {
    marginBottom: 12,
    borderRadius: 12,
    backgroundColor: '#FFFFFF',
    elevation: 2,
  },
  heatmapListWrapper: {
    marginTop: 16,
  },
  heatmapList: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingVertical: 6,
    paddingLeft: 4,
    paddingRight: 12,
  },
  heatmapItem: {
    width: 140,
    marginRight: 16,
  },
  heatmapCanvas: {
    width: 140,
    height: 180,
    borderRadius: 18,
    overflow: 'hidden',
    padding: 16,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1C1537',
  },
  heatmapHighlight: {
    position: 'absolute',
    borderRadius: 120,
  },
  heatmapBloom: {
    position: 'absolute',
    borderRadius: 150,
  },
  heatmapChestGlow: {
    position: 'absolute',
    left: -20,
    right: -20,
    top: 0,
    bottom: 0,
  },
  heatmapStreak: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 2,
    backgroundColor: 'rgba(148, 163, 184, 0.35)',
  },
  heatmapFrame: {
    ...StyleSheet.absoluteFillObject,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.14)',
    borderRadius: 18,
  },
  heatmapCaption: {
    marginTop: 8,
    fontSize: 12,
    fontWeight: '600',
    color: '#1F2937',
  },
  assetFrame: {
    width: 140,
    height: 180,
    borderRadius: 18,
    overflow: 'hidden',
    position: 'relative',
    backgroundColor: '#0F172A',
  },
  assetImage: {
    width: '100%',
    height: '100%',
  },
  assetOverlay: {
    ...StyleSheet.absoluteFillObject,
  },
  assetBorder: {
    ...StyleSheet.absoluteFillObject,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.16)',
    borderRadius: 18,
  },
  emptyScrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 40,
  },
  emptyIcon: {
    marginBottom: 16,
  },
  emptyTitle: {
    color: '#1F2937',
    fontWeight: '600',
    marginBottom: 8,
  },
  emptySubtitle: {
    color: '#9CA3AF',
    textAlign: 'center',
    paddingHorizontal: 20,
  },
  loadingFooter: {
    paddingVertical: 20,
    alignItems: 'center',
  },
  loadingText: {
    color: '#9CA3AF',
    fontSize: 14,
  },
});
