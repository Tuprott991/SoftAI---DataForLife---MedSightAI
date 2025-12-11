import { useTranslation } from 'react-i18next';
import { User, Calendar, Activity, Droplet } from 'lucide-react';

export const PatientInfo = ({ patient }) => {
    const { t } = useTranslation();

    // Format date
    const formatDate = (dateString) => {
        if (!dateString) return 'N/A';
        try {
            const date = new Date(dateString);
            return date.toLocaleDateString('vi-VN', {
                year: 'numeric',
                month: 'long',
                day: 'numeric'
            });
        } catch {
            return dateString;
        }
    };

    return (
        <div className="relative inline-block group">
            <button className="bg-[#2a2a2a] text-white px-4 py-2 rounded-lg shadow-md hover:bg-[#383838] transition">
                {t('doctorDetail.patientInfo.title')}
            </button>

            <div className="absolute right-0 mt-2 w-72 bg-[#2a2a2a] p-4 rounded-lg shadow-lg shadow-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none group-hover:pointer-events-auto z-10">
                <h3 className="text-lg font-bold mb-3 text-teal-400">{t('patientInfo.personalInfo')}</h3>
                
                <div className="space-y-2">
                    <div className="flex items-center gap-2">
                        <User className="w-4 h-4 text-gray-400" />
                        <p className="text-sm text-gray-300">
                            <span className="font-semibold">{t('doctorDetail.patientInfo.name')}:</span> {patient.name}
                        </p>
                    </div>

                    {patient.age && (
                        <div className="flex items-center gap-2">
                            <Calendar className="w-4 h-4 text-gray-400" />
                            <p className="text-sm text-gray-300">
                                <span className="font-semibold">{t('doctor.patientCard.age')}:</span> {patient.age} {t('doctor.patientCard.years')}
                            </p>
                        </div>
                    )}

                    {patient.gender && (
                        <div className="flex items-center gap-2">
                            <User className="w-4 h-4 text-gray-400" />
                            <p className="text-sm text-gray-300">
                                <span className="font-semibold">{t('doctor.patientCard.gender')}:</span> {patient.gender}
                            </p>
                        </div>
                    )}

                    {patient.blood_type && (
                        <div className="flex items-center gap-2">
                            <Droplet className="w-4 h-4 text-gray-400" />
                            <p className="text-sm text-gray-300">
                                <span className="font-semibold">{t('doctor.patientCard.bloodType')}:</span> {patient.blood_type}
                            </p>
                        </div>
                    )}

                    {patient.status && (
                        <div className="flex items-center gap-2">
                            <Activity className="w-4 h-4 text-gray-400" />
                            <p className="text-sm text-gray-300">
                                <span className="font-semibold">{t('doctorDetail.patientInfo.status')}:</span> {patient.status}
                            </p>
                        </div>
                    )}

                    {patient.created_at && (
                        <div className="flex items-center gap-2">
                            <Calendar className="w-4 h-4 text-gray-400" />
                            <p className="text-sm text-gray-300">
                                <span className="font-semibold">{t('doctorDetail.patientInfo.admitted')}:</span> {formatDate(patient.created_at)}
                            </p>
                        </div>
                    )}

                    {patient.underlying_condition && (
                        <div className="mt-3 pt-3 border-t border-gray-700">
                            <p className="text-sm font-semibold text-gray-300 mb-2">{t('doctorDetail.patientInfo.conditions')}:</p>
                            <div className="space-y-1 text-sm text-gray-400">
                                {patient.underlying_condition.hypertension && (
                                    <p>• {t('doctorDetail.patientInfo.hypertension')}</p>
                                )}
                                {patient.underlying_condition.diabetes && (
                                    <p>• {t('doctorDetail.patientInfo.diabetes')}</p>
                                )}
                                {patient.underlying_condition.asthma && (
                                    <p>• {t('doctorDetail.patientInfo.asthma')}</p>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
