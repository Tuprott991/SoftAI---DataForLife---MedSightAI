import React, { useEffect, useMemo, useState } from 'react';
import {
  View,
  ScrollView,
  StyleSheet,
  Image,
  TouchableOpacity,
  ImageSourcePropType,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Text, Card, Button, List, Chip, Avatar, Divider, IconButton } from 'react-native-paper';
import { useNavigation, useRoute } from '@react-navigation/native';
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
      type: 'asset';
      source: ImageSourcePropType;
      label?: string;
    }
  | {
      id: string;
      type: 'heatmap';
      preset: HeatmapPresetKey;
      label?: string;
    };

const HEATMAP_PRESETS: Record<HeatmapPresetKey, HeatmapPresetConfig> = {
  focalLeft: {
    baseColors: ['#160F2B', '#1F1740', '#2B2056'],
    highlight: {
      colors: ['rgba(24, 219, 172, 0.9)', 'rgba(64, 184, 244, 0.85)', 'rgba(248, 94, 94, 0.9)'],
      size: { width: 260, height: 300 },
      position: { top: -70, left: -50 },
      start: { x: 0.12, y: 0.1 },
      end: { x: 0.85, y: 0.9 },
      rotate: '-16deg',
    },
    bloom: {
      colors: ['rgba(20, 184, 166, 0.25)', 'rgba(20, 184, 166, 0)'],
      size: { width: 320, height: 360 },
      position: { top: -100, left: -80 },
      opacity: 0.82,
    },
    streaks: [
      { top: 68, opacity: 0.22 },
      { top: 142, opacity: 0.16 },
      { top: 216, opacity: 0.18 },
    ],
  },
  diffuseCentral: {
    baseColors: ['#160F2B', '#1E183F', '#292053'],
    highlight: {
      colors: ['rgba(88, 247, 195, 0.9)', 'rgba(246, 201, 86, 0.85)', 'rgba(242, 97, 118, 0.8)'],
      size: { width: 340, height: 260 },
      position: { top: -60, left: -110 },
      start: { x: 0.25, y: 0 },
      end: { x: 0.55, y: 1 },
    },
    bloom: {
      colors: ['rgba(246, 201, 86, 0.2)', 'rgba(246, 201, 86, 0)'],
      size: { width: 380, height: 380 },
      position: { top: -100, left: -120 },
      opacity: 0.85,
    },
    streaks: [
      { top: 60, opacity: 0.18 },
      { top: 150, opacity: 0.14 },
      { top: 230, opacity: 0.19 },
    ],
  },
  focalRight: {
    baseColors: ['#160F2B', '#1C1539', '#241C4A'],
    highlight: {
      colors: ['rgba(46, 199, 226, 0.85)', 'rgba(76, 248, 166, 0.9)', 'rgba(248, 112, 112, 0.85)'],
      size: { width: 260, height: 290 },
      position: { top: -74, left: -90 },
      start: { x: 0.2, y: 0.12 },
      end: { x: 0.92, y: 0.88 },
      rotate: '14deg',
    },
    bloom: {
      colors: ['rgba(76, 248, 166, 0.22)', 'rgba(76, 248, 166, 0)'],
      size: { width: 320, height: 340 },
      position: { top: -110, left: -120 },
      opacity: 0.8,
    },
    streaks: [
      { top: 70, opacity: 0.2 },
      { top: 152, opacity: 0.14 },
      { top: 234, opacity: 0.16 },
    ],
  },
  baseline: {
    baseColors: ['#120C24', '#1A1234', '#231945'],
    highlight: {
      colors: ['rgba(70, 235, 210, 0.75)', 'rgba(144, 202, 249, 0.6)', 'rgba(162, 114, 255, 0.6)'],
      size: { width: 240, height: 300 },
      position: { top: -60, left: -40 },
      start: { x: 0.2, y: 0.15 },
      end: { x: 0.6, y: 0.92 },
    },
    streaks: [
      { top: 82, opacity: 0.12 },
      { top: 172, opacity: 0.14 },
      { top: 252, opacity: 0.1 },
    ],
  },
};

export default function ResultDetailScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute<any>();
  const { caseData } = route.params;

  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  const images = useMemo<CasePreviewImage[]>(() => {
    const rawImages = Array.isArray(caseData?.images) ? caseData.images : [];

    return rawImages
      .map((item: any, index: number) => {
        if (item?.type === 'asset' && item.source) {
          return {
            id: item.id ?? `asset_${index}`,
            type: 'asset' as const,
            source: item.source as ImageSourcePropType,
            label: item.label,
          } as CasePreviewImage;
        }

        if (
          item?.type === 'heatmap' &&
          typeof item.preset === 'string' &&
          item.preset in HEATMAP_PRESETS
        ) {
          const preset = item.preset as HeatmapPresetKey;
          return {
            id: item.id ?? `heatmap_${index}`,
            type: 'heatmap' as const,
            preset,
            label: item.label,
          } as CasePreviewImage;
        }

        if (typeof item === 'string' && item.length > 0) {
          return {
            id: `legacy_uri_${index}`,
            type: 'asset' as const,
            source: { uri: item } as ImageSourcePropType,
          } as CasePreviewImage;
        }

        return undefined;
      })
      .filter(
        (entry: CasePreviewImage | undefined): entry is CasePreviewImage => Boolean(entry)
      );
  }, [caseData?.images]);

  const isCompleted = caseData.status === 'completed';
  const hasImages = images.length > 0;
  const currentPreview = hasImages ? images[currentImageIndex] : null;

  useEffect(() => {
    setCurrentImageIndex(0);
  }, [images]);

  const handleNextImage = () => {
    if (currentImageIndex < images.length - 1) {
      setCurrentImageIndex((prev) => prev + 1);
    }
  };

  const handlePrevImage = () => {
    if (currentImageIndex > 0) {
      setCurrentImageIndex((prev) => prev - 1);
    }
  };

  const handleGoBack = () => {
    navigation.goBack();
  };

  const renderHeatmapPreview = (preset: HeatmapPresetKey) => {
    const config = HEATMAP_PRESETS[preset];

    return (
      <View style={styles.detailHeatmapWrap}>
        <LinearGradient colors={config.baseColors} style={styles.detailHeatmapCanvas}>
          <LinearGradient
            colors={config.highlight.colors}
            start={config.highlight.start}
            end={config.highlight.end}
            style={[
              styles.detailHeatmapHighlight,
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
                styles.detailHeatmapBloom,
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
            colors={['rgba(255, 255, 255, 0.12)', 'rgba(255, 255, 255, 0)'] as [string, string]}
            start={{ x: 0.5, y: 0 }}
            end={{ x: 0.5, y: 1 }}
            style={styles.detailHeatmapGlow}
          />

          {config.streaks?.map((streak, index) => (
            <View
              key={`detail-streak-${preset}-${index}`}
              style={[
                styles.detailHeatmapStreak,
                {
                  top: streak.top,
                  opacity: streak.opacity,
                },
              ]}
            />
          ))}

          <View style={styles.detailHeatmapFrame} />
        </LinearGradient>
      </View>
    );
  };

  const renderThumbnail = (preview: CasePreviewImage, index: number) => {
    const isActive = currentImageIndex === index;

    if (preview.type === 'asset') {
      return (
        <Image
          source={preview.source}
          style={[styles.thumbnail, isActive && styles.thumbnailActive]}
          resizeMode="cover"
        />
      );
    }

    const preset = HEATMAP_PRESETS[preview.preset];
    return (
      <LinearGradient
        colors={preset.baseColors}
        style={[styles.thumbnailHeatmap, isActive && styles.thumbnailActive]}
      >
        <View style={styles.thumbnailOverlay} />
      </LinearGradient>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Header with Back Button */}
        <View style={styles.header}>
          <TouchableOpacity onPress={handleGoBack}>
            <IconButton icon="arrow-left" size={28} iconColor="#14B8A6" />
          </TouchableOpacity>
          <Text variant="titleLarge" style={styles.headerTitle}>
            Chi tiết hồ sơ
          </Text>
          <View style={{ width: 40 }} />
        </View>

        {/* Image Section */}
        {hasImages ? (
          <View style={styles.imageSection}>
            <View style={styles.imageContainer}>
              {currentPreview?.type === 'asset' ? (
                <Image source={currentPreview.source} style={styles.image} resizeMode="cover" />
              ) : currentPreview ? (
                renderHeatmapPreview(currentPreview.preset)
              ) : (
                <View style={[styles.previewFallback]}>
                  <Avatar.Icon
                    icon="image-off-outline"
                    size={64}
                    style={{ backgroundColor: '#D1D5DB' }}
                  />
                  <Text style={styles.noImageText}>Không có hình ảnh</Text>
                </View>
              )}

              {images.length > 1 && (
                <>
                  {currentImageIndex > 0 && (
                    <TouchableOpacity
                      style={[styles.imageNavBtn, styles.prevBtn]}
                      onPress={handlePrevImage}
                    >
                      <Text style={styles.navBtnText}>‹</Text>
                    </TouchableOpacity>
                  )}

                  {currentImageIndex < images.length - 1 && (
                    <TouchableOpacity
                      style={[styles.imageNavBtn, styles.nextBtn]}
                      onPress={handleNextImage}
                    >
                      <Text style={styles.navBtnText}>›</Text>
                    </TouchableOpacity>
                  )}
                </>
              )}
            </View>

            {currentPreview?.label ? (
              <Text style={styles.imageCaption}>{currentPreview.label}</Text>
            ) : null}

            {images.length > 1 && (
              <Text style={styles.imageCounter}>
                {currentImageIndex + 1} / {images.length}
              </Text>
            )}

            {images.length > 1 && (
              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                style={styles.thumbnailContainer}
              >
                {images.map((preview, index) => (
                  <TouchableOpacity
                    key={preview.id}
                    onPress={() => setCurrentImageIndex(index)}
                    activeOpacity={0.8}
                  >
                    {renderThumbnail(preview, index)}
                  </TouchableOpacity>
                ))}
              </ScrollView>
            )}
          </View>
        ) : (
          <View style={styles.imageSection}>
            <View style={[styles.imageContainer, styles.noImagePlaceholder]}>
              <Avatar.Icon
                icon="image-off-outline"
                size={64}
                style={{ backgroundColor: '#D1D5DB' }}
              />
              <Text style={styles.noImageText}>Không có hình ảnh</Text>
            </View>
          </View>
        )}

        {/* Content Section */}
        <View style={styles.content}>
          {/* Case Info */}
          <Card style={styles.infoCard}>
            <Card.Title
              title="Thông tin hồ sơ"
              titleStyle={{ color: '#1F2937', fontWeight: 'bold' }}
              left={(props) => <List.Icon {...props} icon="information-outline" />}
            />
            <Card.Content>
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Mã hồ sơ:</Text>
                <Text style={styles.infoValue}>{caseData.id}</Text>
              </View>
              <Divider style={styles.divider} />
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Ngày khám:</Text>
                <Text style={styles.infoValue}>{caseData.created_at}</Text>
              </View>
              <Divider style={styles.divider} />
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Trạng thái:</Text>
                <Chip
                  icon={isCompleted ? 'check-circle' : 'clock-outline'}
                  style={{
                    backgroundColor: isCompleted ? '#D1FAE5' : '#FEF3C7',
                  }}
                  textStyle={{
                    color: isCompleted ? '#065F46' : '#78350F',
                    fontWeight: '600',
                  }}
                >
                  {isCompleted ? 'Đã hoàn thành' : 'Đang chờ'}
                </Chip>
              </View>
            </Card.Content>
          </Card>

          {/* Symptoms */}
          <Card style={styles.symptomsCard}>
            <Card.Title
              title="Triệu chứng ban đầu"
              titleStyle={{ color: '#1F2937', fontWeight: 'bold' }}
              left={(props) => <List.Icon {...props} icon="hospital-box" />}
            />
            <Card.Content>
              <Text variant="bodyMedium" style={styles.symptomsText}>
                {caseData.symptoms}
              </Text>
            </Card.Content>
          </Card>

          {/* AI Diagnosis */}
          {isCompleted && caseData.doctor_diagnosis && (
            <>
              <Card style={[styles.resultCard, styles.aiCard]}>
                <Card.Title
                  title="Gợi ý từ AI (MedSight)"
                  titleStyle={{ color: '#0369A1', fontWeight: 'bold' }}
                  left={(props) => <List.Icon {...props} icon="robot" color="#0369A1" />}
                />
                <Card.Content>
                  <Text variant="bodyLarge" style={styles.predictionText}>
                    <Text style={{ fontWeight: 'bold' }}>Dự đoán:</Text>{' '}
                    {caseData.doctor_diagnosis.ai_prediction}
                  </Text>
                  <Text variant="bodySmall" style={styles.confidenceText}>
                    (Mức độ tin cậy: Cao)
                  </Text>
                </Card.Content>
              </Card>

              {/* Doctor Diagnosis */}
              <Card style={[styles.resultCard, styles.doctorCard]}>
                <Card.Title
                  title="Kết luận bác sĩ"
                  titleStyle={{ color: '#059669', fontWeight: 'bold' }}
                  left={(props) => <List.Icon {...props} icon="doctor" color="#059669" />}
                  subtitle={`Dr. ${caseData.doctor_diagnosis.doctor_name}`}
                  subtitleStyle={{ fontSize: 12 }}
                />
                <Divider />
                <Card.Content>
                  <Text variant="labelLarge" style={styles.sectionLabel}>
                    Kết luận:
                  </Text>
                  <Text
                    variant="bodyMedium"
                    style={[styles.diagnosisText, { fontWeight: 'bold', color: '#059669' }]}
                  >
                    {caseData.doctor_diagnosis.conclusion}
                  </Text>

                  <Text variant="labelLarge" style={[styles.sectionLabel, { marginTop: 12 }]}>Ghi chú:</Text>
                  <Text variant="bodyMedium" style={styles.diagnosisText}>
                    {caseData.doctor_diagnosis.notes}
                  </Text>
                </Card.Content>
              </Card>

              {/* Recommendations */}
              <Card style={styles.recommendationCard}>
                <Card.Title
                  title="Hướng dẫn tiếp theo"
                  titleStyle={{ color: '#1F2937', fontWeight: 'bold' }}
                  left={(props) => (
                    <List.Icon {...props} icon="clipboard-check-outline" />
                  )}
                />
                <Card.Content>
                  <View style={styles.recommendationItem}>
                    <Text style={styles.recommendationBullet}>•</Text>
                    <Text style={styles.recommendationText}>
                      Tuân thủ theo hướng dẫn của bác sĩ
                    </Text>
                  </View>
                  <View style={styles.recommendationItem}>
                    <Text style={styles.recommendationBullet}>•</Text>
                    <Text style={styles.recommendationText}>
                      Liên hệ bác sĩ nếu có triệu chứng bất thường
                    </Text>
                  </View>
                  <View style={styles.recommendationItem}>
                    <Text style={styles.recommendationBullet}>•</Text>
                    <Text style={styles.recommendationText}>
                      Tái khám theo lịch hẹn
                    </Text>
                  </View>
                </Card.Content>
              </Card>
            </>
          )}

          {/* Pending State */}
          {!isCompleted && (
            <Card style={styles.pendingCard}>
              <Card.Content>
                <View style={styles.pendingContent}>
                  <Avatar.Icon
                    icon="hourglass-empty"
                    size={64}
                    style={{ backgroundColor: '#FEF3C7' }}
                  />
                  <Text
                    variant="headlineSmall"
                    style={[styles.pendingTitle, { marginTop: 16 }]}
                  >
                    Đang chờ bác sĩ
                  </Text>
                  <Text variant="bodyMedium" style={styles.pendingSubtitle}>
                    Bác sĩ sẽ sớm kiểm tra và phản hồi hồ sơ của bạn
                  </Text>
                </View>
              </Card.Content>
            </Card>
          )}

          {/* Action Buttons */}
          <View style={styles.actionButtons}>
            <Button
              mode="outlined"
              onPress={handleGoBack}
              style={styles.actionBtn}
              textColor="#14B8A6"
            >
              Quay lại
            </Button>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F7FA',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 8,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  headerTitle: {
    color: '#1F2937',
    fontWeight: '600',
    flex: 1,
    textAlign: 'center',
  },
  imageSection: {
    backgroundColor: '#FFFFFF',
    paddingVertical: 12,
  },
  imageContainer: {
    position: 'relative',
    height: 300,
    backgroundColor: '#000000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  previewFallback: {
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    height: '100%',
  },
  imageCaption: {
    textAlign: 'center',
    marginTop: 10,
    fontSize: 14,
    color: '#E5E7EB',
    fontWeight: '600',
  },
  noImagePlaceholder: {
    backgroundColor: '#F3F4F6',
  },
  noImageText: {
    color: '#9CA3AF',
    fontSize: 16,
    marginTop: 12,
  },
  imageNavBtn: {
    position: 'absolute',
    width: 44,
    height: 44,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
  },
  prevBtn: {
    left: 12,
  },
  nextBtn: {
    right: 12,
  },
  navBtnText: {
    color: '#FFFFFF',
    fontSize: 32,
    fontWeight: 'bold',
  },
  imageCounter: {
    textAlign: 'center',
    marginVertical: 8,
    color: '#6B7280',
    fontSize: 12,
    fontWeight: '600',
  },
  thumbnailContainer: {
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  thumbnail: {
    width: 60,
    height: 60,
    borderRadius: 8,
    marginRight: 8,
    borderWidth: 2,
    borderColor: '#E5E7EB',
  },
  thumbnailHeatmap: {
    width: 60,
    height: 60,
    borderRadius: 8,
    marginRight: 8,
    borderWidth: 2,
    borderColor: '#E5E7EB',
    overflow: 'hidden',
    justifyContent: 'center',
    alignItems: 'center',
  },
  thumbnailOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(17, 24, 39, 0.2)',
  },
  thumbnailActive: {
    borderColor: '#14B8A6',
  },
  detailHeatmapWrap: {
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  detailHeatmapCanvas: {
    width: '88%',
    height: '88%',
    borderRadius: 28,
    overflow: 'hidden',
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
    backgroundColor: '#0F172A',
  },
  detailHeatmapHighlight: {
    position: 'absolute',
  },
  detailHeatmapBloom: {
    position: 'absolute',
  },
  detailHeatmapGlow: {
    ...StyleSheet.absoluteFillObject,
  },
  detailHeatmapStreak: {
    position: 'absolute',
    left: 18,
    right: 18,
    height: 2,
    backgroundColor: 'rgba(148, 163, 184, 0.35)',
  },
  detailHeatmapFrame: {
    ...StyleSheet.absoluteFillObject,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.18)',
    borderRadius: 28,
  },
  content: {
    padding: 12,
  },
  infoCard: {
    marginBottom: 12,
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  infoLabel: {
    color: '#6B7280',
    fontSize: 14,
  },
  infoValue: {
    color: '#1F2937',
    fontWeight: '600',
  },
  divider: {
    marginVertical: 8,
    backgroundColor: '#E5E7EB',
  },
  symptomsCard: {
    marginBottom: 12,
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
  },
  symptomsText: {
    color: '#4B5563',
    lineHeight: 20,
  },
  resultCard: {
    marginBottom: 12,
    borderRadius: 12,
  },
  aiCard: {
    backgroundColor: '#F0F9FF',
  },
  doctorCard: {
    backgroundColor: '#ECFDF5',
  },
  predictionText: {
    color: '#0C4A6E',
  },
  confidenceText: {
    color: '#0C4A6E',
    marginTop: 4,
  },
  sectionLabel: {
    color: '#1F2937',
    marginBottom: 4,
  },
  diagnosisText: {
    color: '#374151',
    lineHeight: 20,
  },
  recommendationCard: {
    marginBottom: 12,
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
  },
  recommendationItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 6,
  },
  recommendationBullet: {
    color: '#14B8A6',
    fontSize: 18,
    lineHeight: 20,
    marginRight: 6,
  },
  recommendationText: {
    color: '#374151',
    flex: 1,
    lineHeight: 20,
  },
  pendingCard: {
    marginBottom: 12,
    backgroundColor: '#FFFBEB',
    borderRadius: 12,
  },
  pendingContent: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 20,
  },
  pendingTitle: {
    color: '#B45309',
    fontWeight: '600',
  },
  pendingSubtitle: {
    color: '#92400E',
    textAlign: 'center',
    paddingHorizontal: 20,
  },
  actionButtons: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 12,
  },
  actionBtn: {
    borderColor: '#14B8A6',
  },
});
