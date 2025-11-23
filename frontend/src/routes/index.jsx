import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { Layout } from '../components/layout';
import { Home } from './Home';
import { Doctor } from './Doctor';
import { Student } from './Student';
import { DoctorDetail } from './DoctorDetail';
import { StudentDetail } from './StudentDetail';
import { NotFound } from './NotFound';

const router = createBrowserRouter([
    {
        path: '/',
        element: <Layout />,
        children: [
            {
                path: '/',
                element: <Home />
            },
            {
                path: '/home',
                element: <Home />
            },
            {
                path: '/doctor',
                element: <Doctor />
            },
            {
                path: '/doctor/:id',
                element: <DoctorDetail />
            },
            {
                path: '/student',
                element: <Student />
            },
            {
                path: '/student/:id',
                element: <StudentDetail />
            },
            {
                path: '*',
                element: <NotFound />
            }
        ]
    }
]);

export const AppRouter = () => {
    return <RouterProvider router={router} />;
};
