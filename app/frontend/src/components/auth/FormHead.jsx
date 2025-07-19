import Link from '@mui/material/Link';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { RouterLink } from "../../routes/RouterLink";

const FormFields = ({ sx, title, variant, href }) => {
    return (
      <Stack spacing={1.5} sx={sx}>
        <Typography variant="h4">{title}</Typography>
  
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          {variant === 'sign-in' ? 'Belum punya akun? ' : 'Sudah punya akun? '}
  
          <Link component={RouterLink} href={href} variant="subtitle2">
            {variant === 'sign-in' ? 'Buat akun' : 'Masuk'}
          </Link>
        </Typography>
      </Stack>
    );
    
  }

  export default FormFields;