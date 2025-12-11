/**
 * Map diagnosis text (Vietnamese or English) to translation key
 * @param {string} diagnosis - Diagnosis text in Vietnamese or English
 * @returns {string} Translation key
 */
export const getDiagnosisKey = (diagnosis) => {
  const diagnosisMap = {
    // Vietnamese
    "Lao phổi": "diagnosis.tuberculosis",
    "Viêm phổi": "diagnosis.pneumonia",
    "Bệnh phổi khác": "diagnosis.otherLungDisease",
    "Chưa phát hiện": "diagnosis.noFinding",
    // English
    Tuberculosis: "diagnosis.tuberculosis",
    Pneumonia: "diagnosis.pneumonia",
    "Other Lung Disease": "diagnosis.otherLungDisease",
    "No Finding": "diagnosis.noFinding",
  };

  return diagnosisMap[diagnosis] || diagnosis;
};

/**
 * Get translated diagnosis text
 * @param {string} diagnosis - Vietnamese diagnosis text
 * @param {Function} t - Translation function from useTranslation
 * @returns {string} Translated diagnosis
 */
export const getTranslatedDiagnosis = (diagnosis, t) => {
  const key = getDiagnosisKey(diagnosis);
  // If key starts with 'diagnosis.', it's a translation key
  if (key.startsWith("diagnosis.")) {
    return t(key);
  }
  // Otherwise return as is
  return diagnosis;
};

/**
 * Get translated gender text
 * @param {string} gender - Gender text ("Nam", "Nữ", "Male", "Female")
 * @param {Function} t - Translation function from useTranslation
 * @returns {string} Translated gender
 */
export const getTranslatedGender = (gender, t) => {
  const genderLower = gender?.toLowerCase();
  if (genderLower === "nam" || genderLower === "male") {
    return t("doctorDetail.patientInfo.male");
  } else if (genderLower === "nữ" || genderLower === "female") {
    return t("doctorDetail.patientInfo.female");
  }
  return gender;
};

/**
 * Get translated status text
 * @param {string} status - Status from API ("critical", "stable", "improving", "admitted")
 * @param {Function} t - Translation function from useTranslation
 * @returns {string} Translated status
 */
export const getTranslatedStatus = (status, t) => {
  const statusLower = status?.toLowerCase();
  switch (statusLower) {
    case 'critical':
      return t('doctorDetail.patientInfo.critical'); // Nguy kịch
    case 'improving':
      return t('doctorDetail.patientInfo.improving'); // Đang điều trị
    case 'stable':
      return t('doctorDetail.patientInfo.stable'); // Ổn định
    case 'admitted':
      return t('doctorDetail.patientInfo.admitted'); // Tiếp nhận
    default:
      return status;
  }
};

/**
 * Get status color class
 * @param {string} status - Status from API
 * @returns {string} Tailwind CSS classes for status badge
 */
export const getStatusColor = (status) => {
  const statusLower = status?.toLowerCase();
  switch (statusLower) {
    case 'critical':
      return 'bg-red-600/30 text-white border-red-600/40'; // 🔴 Nguy kịch
    case 'improving':
      return 'bg-blue-500/30 text-white border-blue-500/40'; // 🔵 Đang điều trị
    case 'stable':
      return 'bg-green-500/30 text-white border-green-500/40'; // 🟢 Ổn định
    case 'admitted':
      return 'bg-teal-500/30 text-white border-teal-500/40'; // 🔷 Tiếp nhận
    default:
      return 'bg-gray-500/30 text-white border-gray-500/40';
  }
};
