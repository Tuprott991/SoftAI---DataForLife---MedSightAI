const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Get list of patients with pagination
 * @param {number} page - Page number (default: 1)
 * @param {number} pageSize - Number of items per page (default: 20)
 * @returns {Promise<Object>} - Response containing patients data
 */
export const getPatients = async (page = 1, pageSize = 20) => {
    try {
        const response = await fetch(
            `${API_BASE_URL}/api/v1/patients/?page=${page}&page_size=${pageSize}`
        );
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching patients:', error);
        throw error;
    }
};

/**
 * Get detailed information of a specific patient
 * @param {string} patientId - Patient ID (UUID)
 * @returns {Promise<Object>} - Patient detailed information
 */
export const getPatientDetail = async (patientId) => {
    try {
        const response = await fetch(
            `${API_BASE_URL}/api/v1/patients/${patientId}/infor`
        );
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching patient detail:', error);
        throw error;
    }
};

/**
 * Convert DICOM URL to displayable image URL
 * @param {string} dicomUrl - DICOM file URL from S3
 * @returns {string} - Converted image URL for display
 */
export const getDicomImageUrl = (dicomUrl) => {
    if (!dicomUrl) return null;
    
    // If it's already a processed image, return as is
    if (dicomUrl.includes('.png') || dicomUrl.includes('.jpg') || dicomUrl.includes('.jpeg')) {
        return dicomUrl;
    }
    
    // For DICOM files, we'll need to use a backend endpoint to convert
    // The backend should provide an endpoint like: /api/v1/dicom/view?url=<dicom_url>
    const encodedUrl = encodeURIComponent(dicomUrl);
    return `${API_BASE_URL}/api/v1/dicom/view?url=${encodedUrl}`;
};

/**
 * Search patients by name
 * @param {string} searchQuery - Search query string
 * @param {number} page - Page number
 * @param {number} pageSize - Number of items per page
 * @returns {Promise<Object>} - Filtered patients data
 */
export const searchPatients = async (searchQuery, page = 1, pageSize = 20) => {
    try {
        const response = await fetch(
            `${API_BASE_URL}/api/v1/patients/?page=${page}&page_size=${pageSize}&search=${encodeURIComponent(searchQuery)}`
        );
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error searching patients:', error);
        throw error;
    }
};
